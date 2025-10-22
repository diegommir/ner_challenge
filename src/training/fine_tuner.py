import os
import pandas as pd
import torch

from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def encode_category(category):
    if category == 'numbers':
        return 1
    if category == 'unusual':
        return 2
    if category == 'non_english':
        return 3
    else:
        return 0

class CustomDataset(Dataset):
    def __init__(self, questions: list[str], categories: list[int], tokenizer: BertTokenizer, token_length: int):
        self.questions = questions
        self.categories = categories
        self.tokenizer = tokenizer
        self.token_length = token_length

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, i):
        tokenized = self.tokenizer(
            self.questions[i],
            max_length=self.token_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'categories': torch.tensor(self.categories[i], dtype=torch.long)
        }

def run_training():
    file_path = os.path.join(os.path.dirname(__file__), '../../files/train_set.csv')
    df = pd.read_csv(file_path)
    df['category_encoded'] = df['category'].map(encode_category)
    print(df)

    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')

    token_length = 128
    train_set = CustomDataset(df['question'].values, df['category_encoded'].values, tokenizer, token_length)
    # validation_set = CustomDataset(df['question'].values, df['category_encoded'].values, tokenizer, token_length)

    batch_size = 1
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # validation_dl = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    # Training setup
    device = torch.device('cpu')
    model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-uncased', num_labels=3)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Start Training
    epochs = 2
    for epoch in range(epochs):
        print('Epoch #', epoch, 'starting...')

        # Set model for training mode
        model.train()
        total_loss = 0

        # tqdm for showing a progress bar. Not really necessary.
        loop = tqdm(train_dl)

        for batch in loop:
            # Zeroing gradient 
            optimizer.zero_grad()

            # Get batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            categories = batch['categories'].to(device)

            # Generate prediction
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=categories)

            # Get loss info
            loss = output.loss
            total_loss += loss.item()

            # Apply backward propagation
            loss.backward()

            # Update model params
            optimizer.step()

            # Show loss info on the progress bar
            loop.set_postfix(loss=loss.item())

        print('Epoch #', epoch, 'end. Loss:', total_loss / len(train_dl))

    # Start Evaluation
    model.eval()

    with torch.no_grad():
        it = iter(train_dl)
        item = next(it)

        input_ids = item['input_ids'].to(device)
        attention_mask = item['attention_mask'].to(device)
        categories = item['categories'].to(device)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(output.logits, dim=1)

        print('Pred:', pred, 'Cat:', categories)
        
        item = next(it)

        input_ids = item['input_ids'].to(device)
        attention_mask = item['attention_mask'].to(device)
        categories = item['categories'].to(device)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(output.logits, dim=1)

        print('Pred:', pred, 'Cat:', categories)

if __name__ == '__main__':
    print('Command line run...')
    run_training()

import os
import pandas as pd
import torch

from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def encode_category(category):
    '''This function encode/converts the categories from strings to numbers, 
    in a way that the model can be trained'''
    if category == 'numbers':
        return 1
    if category == 'unusual':
        return 2
    if category == 'non_english':
        return 3
    else:
        return 0

class CustomDataset(Dataset):
    '''This class encapsulate the dataset that is going to be used during training.
    It also returns the dataset values already tokenized and ready to be used by the model.'''
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

def run_training(relative_path: str):
    '''This function trains the model, evaluates it and saves it to the disk for later use.'''
    print('Starting training...')

    # Loads the training dataset and apply de encoding of the labels.
    file_path = os.path.join(os.path.dirname(__file__), relative_path)
    df = pd.read_csv(file_path)
    df['category_encoded'] = df['category'].map(encode_category)
    # print(df.head())

    # This is used to transform data into tokens, which is used by the model.
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')

    # Splitting training dataset into two different sets in order to test que quality of the training.
    # In this case 70% of the data goes to training and 30% goes to evaluation.
    # Random state was set fixed in to assess the impact of changes on hyperparams.
    x_train, x_test, y_train, y_test = train_test_split(df['question'].values, df['category_encoded'].values, train_size=0.7, random_state=42)

    # This is a technicality of Torch. This is a common way to pass data to the model using Torch.
    # So first we create our custom dataset....
    token_length = 128
    train_set = CustomDataset(x_train, y_train, tokenizer, token_length)
    test_set = CustomDataset(x_test, y_test, tokenizer, token_length)

    # ... and then we create a DataLoader, which is responsible to iterate through the data during test and eval steps.
    batch_size = 10
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # Training setup
    device_name = 'cpu'
    if torch.cuda.is_available():
        device_name = 'cuda'
    print('Device Name:', device_name)
    device = torch.device(device_name)
    model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-uncased', num_labels=2)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=0.00005)

    # Start Training
    print('Training loop...')
    epochs = 3
    for epoch in range(epochs):
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
    print('Evaluation loop...')
    true_positive = 0
    total = 0

    model.eval()
    with torch.no_grad():

        # tqdm for showing a progress bar. Not really necessary.
        loop = tqdm(test_dl)

        for batch in loop:
            # Get batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            categories = batch['categories'].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(output.logits, dim=1)

            true_positive += (pred == categories).sum().item()
            total += categories.size(0)
        
    accuracy = true_positive / total
    print('Accuracy:', accuracy)
    
    # This is the "torch way" of saving the model for reuse. Since we are going to use transformers
    # along the way, so is not the best way to save it.
    # file_path = os.path.join(os.path.dirname(__file__), f'../../models/bert_questions.pt')
    # torch.save(model.state_dict(), file_path)

    # Transformers way to save the model
    dir_path = os.path.join(os.path.dirname(__file__), f'../../models/bert_questions/')
    model.save_pretrained(dir_path)

if __name__ == '__main__':
    print('Command line run...')
    run_training('../../files/train_set.csv')

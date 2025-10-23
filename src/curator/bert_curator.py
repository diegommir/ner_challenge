import os
import torch
import json

from transformers import BertTokenizer, BertForSequenceClassification

def curate_questions(num_questions: int):
    '''This function applies the trained model to select suitable questions of each category
    based on the given "num_questions" param.'''
    print('Start curating questions...')
    
    print('Loading model...')
    # Set torch as CPU since this time is supposed to run in a simple prod container
    device = torch.device('cpu')

    # Load model and set wights
    dir_path = os.path.join(os.path.dirname(__file__), f'../../models/bert_questions/')
    model = BertForSequenceClassification.from_pretrained(dir_path)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')

    # Curated lists
    numbers_questions = []

    print('Opening questions file...')
    # Open the questions file
    file_path = os.path.join(os.path.dirname(__file__), '../../files/JEOPARDY_QUESTIONS1.json')
    with open(file_path) as file:
        questions = json.load(file)

        print('Classifying questions...')
        # Start classifying questions
        for question in questions:
            question_text = question['question'].strip("'").replace(',', '')

            # Turn question text to tokens
            tokenized = tokenizer(
                question_text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Get token data
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)

            # Predict using trained model
            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                pred = torch.argmax(output.logits, dim=1)

            # If it is numbers category and numbers question still needed.
            if pred.int() == 1 and len(numbers_questions) < num_questions:
                numbers_questions.append(question)
        
            # If has reached the correct num of questions for each category, stop looking
            if len(numbers_questions) >= num_questions:
                break
    
    print('Writing questions to files...')
    # Write the numbers questions to a file
    file_path = os.path.join(os.path.dirname(__file__), '../../files/numbers.json')
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(numbers_questions, file, indent=4)
    
    print('Successfully finished.')

if __name__ == '__main__':
    curate_questions(1000)

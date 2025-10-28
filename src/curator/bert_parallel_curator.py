import os
import re
import json
import torch
import random

from transformers import BertTokenizer, BertForSequenceClassification

def curate_questions(num_questions: int):
    '''This function applies the trained model to select suitable questions of each category
    based on the given "num_questions" param and the "curated_cat".'''
    print('Start curating questions...')
    
    print('Loading model...')
    device_name = 'cpu'
    if torch.cuda.is_available():
        device_name = 'cuda'
    print('Device Name:', device_name)
    device = torch.device(device_name)

    # Load trained models
    numbers_model = load_model('numbers_bert')
    unusual_model = load_model('unusual_bert')
    non_english_model = load_model('non_english_bert')

    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')

    # Curated lists
    numbers_questions = []
    unusual_questions = []
    non_english_questions = []

    print('Opening questions file...')
    # Open the questions file
    file_path = os.path.join(os.path.dirname(__file__), '../../files/JEOPARDY_QUESTIONS1.json')
    with open(file_path) as file:
        questions = json.load(file)
        print('Total Questions:', len(questions))

        print('Classifying questions...')
        # If has reached the correct num of questions for each category, stop looking
        while len(numbers_questions) < num_questions or len(unusual_questions) < num_questions or len(non_english_questions) < num_questions:
            # Get a random question
            question = questions[random.randint(0, len(questions) - 1)]

            # Clean the question text
            question_text = question['question'].strip("'").replace(',', '')
            question_text = re.sub(r'<[^>]+>', '', question_text)

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

            # Predict using trained models
            do_predict(numbers_model, input_ids, attention_mask, question, numbers_questions, num_questions)
            do_predict(unusual_model, input_ids, attention_mask, question, unusual_questions, num_questions)
            do_predict(non_english_model, input_ids, attention_mask, question, non_english_questions, num_questions)

    print('Writing questions to files...')
    # Write the questions to a files
    save_questions(numbers_questions, 'numbers')
    save_questions(unusual_questions, 'unusual')
    save_questions(non_english_questions, 'non_english')
    
    print('Successfully finished.')

def load_model(model_name: str) -> BertForSequenceClassification:
    '''This function loads a model based on the given model_name.'''
    dir_path = os.path.join(os.path.dirname(__file__), f'../../models/{model_name}/')
    model = BertForSequenceClassification.from_pretrained(dir_path)
    model.eval()

    return model

def do_predict(model: BertForSequenceClassification, input_ids: any, attention_mask: any, question: any, curated_questions: list, num_questions: int):
    '''This function do the prediction based on given model and add to the curated_questions list if it still needed.'''
    # Predict using trained model
    with torch.no_grad():
        # Classify for the category
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(output.logits, dim=1)

        # If it is from a curated category and question still needed.
        if pred.int() == 1 and len(curated_questions) < num_questions:
            curated_questions.append(question)

def save_questions(curated_questions: list, category: str):
    '''This function saves a curated_questions list to a JSON file.'''
    file_path = os.path.join(os.path.dirname(__file__), f'../../files/curated_{category}.json')
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(curated_questions, file, indent=4)

if __name__ == '__main__':
    import time
    ini_time = time.time()

    num_questions = 1000
    curate_questions(num_questions)

    exec_time = time.time() - ini_time
    print(f'Execution Time: {exec_time:.6f}')

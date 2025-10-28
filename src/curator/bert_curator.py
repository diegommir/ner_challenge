import os
import re
import json
import torch
import random

from transformers import BertTokenizer, BertForSequenceClassification

def curate_questions(num_questions: int, curated_cat: str):
    '''This function applies the trained model to select suitable questions of each category
    based on the given "num_questions" param and the "curated_cat".'''
    print('Start curating', curated_cat, 'questions...')
    
    print('Loading model...')
    device_name = 'cpu'
    if torch.cuda.is_available():
        device_name = 'cuda'
    print('Device Name:', device_name)
    device = torch.device(device_name)

    # Load model and set wights
    dir_path = os.path.join(os.path.dirname(__file__), f'../../models/{curated_cat}_bert/')
    model = BertForSequenceClassification.from_pretrained(dir_path)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')

    # Curated lists
    curated_questions = []

    print('Opening questions file...')
    # Open the questions file
    file_path = os.path.join(os.path.dirname(__file__), '../../files/JEOPARDY_QUESTIONS1.json')
    with open(file_path) as file:
        questions = json.load(file)
        print('Total Questions:', len(questions))

        # Used to calculate how many iterations needed to find the amount of questions
        total_iter = 0

        print('Classifying questions...')
        # If has reached the correct num of questions for each category, stop looking
        while len(curated_questions) < num_questions:
            # Get a random question
            question = questions[random.randint(0, len(questions) - 1)]

            total_iter += 1

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

            # Predict using trained model
            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                pred = torch.argmax(output.logits, dim=1)

            # If it is from a curated category and question still needed.
            if pred.int() == 1 and len(curated_questions) < num_questions:
                curated_questions.append(question)
    
    print('Writing questions to files...')
    # Write the numbers questions to a file
    file_path = os.path.join(os.path.dirname(__file__), f'../../files/curated_{curated_cat}.json')
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(curated_questions, file, indent=4)
    
    print('Successfully finished.')
    
    ratio = num_questions / total_iter
    print(f'Curated to Total Questions ratio for {curated_cat} category: {ratio:.3f}')
    print('========================================================================')

if __name__ == '__main__':
    import time
    ini_time = time.time()

    num_questions = 1000
    curate_questions(num_questions, 'numbers')
    curate_questions(num_questions, 'unusual')
    curate_questions(num_questions, 'non_english')

    exec_time = time.time() - ini_time
    print(f'Execution Time: {exec_time:.6f}')

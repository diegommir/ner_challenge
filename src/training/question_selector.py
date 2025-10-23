import os
import re
import json

def get_questions(num_questions: int):
    '''This function select "num_questions" of each category to be used during the model training.'''
    numbers_questions = []
    normal_questions = []

    # Open the questions file
    file_path = os.path.join(os.path.dirname(__file__), '../../files/JEOPARDY_QUESTIONS1.json')
    with open(file_path) as file:
        questions = json.load(file)

        for question in questions:
            question_text = question['question'].strip("'").replace(',', '')

            # If has numbers...
            if re.search(r'\d+', question_text):
                # ... and numbers question still needed.
                if len(numbers_questions) < num_questions:
                    numbers_questions.append(f'"{question_text}",numbers\n')
            
            # If it is none of the others and normal ones are still needed.
            elif len(normal_questions) < num_questions:
                normal_questions.append(f'"{question_text}",normal\n')
            
            # If has reached the correct num of questions for each category, stop looking
            if len(numbers_questions) >= num_questions and len(normal_questions) >= num_questions:
                break
    
    # Write the selected questions to a file
    file_path = os.path.join(os.path.dirname(__file__), '../../files/train_set.csv')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('question,category\n')
        file.writelines(numbers_questions)
        file.writelines(normal_questions)

if __name__ == '__main__':
    print('Command line run...')
    get_questions(100)

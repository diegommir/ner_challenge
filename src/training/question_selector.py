import os
import re
import json

def get_questions(num_questions: int, training_cat: str):
    '''This function select "num_questions" of each category to be used during the model training.'''
    print('Selecting', num_questions, 'questions of', training_cat, 'category...')
    cat_questions = []
    normal_questions = []

    # Loads english words dict only if it is not running for numbers category
    english_dict = []
    if training_cat != 'numbers':
        english_dict = get_english_words()

    # Open the questions file
    file_path = os.path.join(os.path.dirname(__file__), '../../files/JEOPARDY_QUESTIONS1.json')
    with open(file_path) as file:
        questions = json.load(file)

        for question in questions:
            # Clean the question text
            question_text = question['question'].strip("'").replace(',', '')
            question_text = re.sub(r'<[^>]+>', '', question_text)

            # if we are selecting numbers category
            if training_cat == 'numbers':
                # If has numbers...
                if re.search(r'\d+', question_text):
                    # ... and numbers question still needed.
                    if len(cat_questions) < num_questions:
                        cat_questions.append(f'"{question_text}",{training_cat}\n')

                # If it is not numbers category and normal ones are still needed.
                elif len(normal_questions) < num_questions:
                    normal_questions.append(f'"{question_text}",normal\n')

            # if cat is either unusual or non_english
            else:
                # If has non english or unusual words...

                # This flag is used to check if the question fell into any category.
                # If not, it means that is a normal category.
                any_cat = False

                # removing numbers
                text_without_nums = re.sub(r'\d+', '', question_text)
                # get a list of words to check
                words = re.findall(r'\b\w+\b', text_without_nums)
                for word in words:
                    # check word against a english words dict
                    # if it is a english word, just go check the next
                    if word.lower() in english_dict:
                        continue

                    # ... and non english words question still needed.
                    if training_cat == 'non_english' and len(cat_questions) < num_questions:
                        any_cat = True
                        cat_questions.append(f'"{question_text}",{training_cat}\n')
                        break
                
                    # ... and unusual words question still needed.
                    # I'm going to consider "unusual proper nouns" any word that are non_english 
                    # and was written with a first capital letter (camel/title case)
                    if training_cat == 'unusual' and word.istitle() and len(cat_questions) < num_questions:
                        any_cat = True
                        cat_questions.append(f'"{question_text}",{training_cat}\n')
                        break

                # If it is none of the others and normal ones are still needed.
                if not any_cat and len(normal_questions) < num_questions:
                    normal_questions.append(f'"{question_text}",normal\n')

            # If has reached the correct num of questions for each category, stop looking
            if len(cat_questions) >= num_questions and len(normal_questions) >= num_questions:
                break
    
    # Write the selected questions to a file
    file_path = os.path.join(os.path.dirname(__file__), f'../../files/{training_cat}_train_set.csv')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('question,category\n')
        file.writelines(cat_questions)
        file.writelines(normal_questions)

def get_english_words():
    file_path = os.path.join(os.path.dirname(__file__), '../../files/words_alpha.txt')
    with open(file_path, 'r') as file:
        word_list = [line.strip() for line in file.readlines()]

    return word_list

if __name__ == '__main__':
    print('Command line run...')
    num_questions = 200
    get_questions(num_questions, 'numbers')
    get_questions(num_questions, 'unusual')
    get_questions(num_questions, 'non_english')

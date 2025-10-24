'''
This module is an attempt to use a larger LLM to help tagging questions properly to be used as training datasets for fine tune a model.
But even for this purpose, it has proven difficult to write a prompt able to describe the characteristics of the problem.
So so since the results are not good, this module should not be used. I'm keeping it here just as one example of utter failure. =)
'''

import os
import json
import random
from openai import OpenAI

# json library is used here to read an write files that are in this format.
# random library is used to get random questions of the dataset
# openai library is used to access Ollama api, that has a compatibility layer with this lib


prompt_template = '''# Instructions
- You are an expert in analyzing and classifying questions.
- The Question will always be in English.
- The Question **can** fit into three categories:
  - unusual - It is reserved for a Question that contain in it a Name of a Person that is considered Unusual. Example: Mr. Potato.
    - The name of the person should be on the question itself.
    - You should not consider names of companies, places, etc. Use the context of the question to decide if it is a Person's name or not.
    - The simple fact that this name is in a foreign language is not considered Unusual.
  - numbers - It is reserved for a Question that contain Numbers. It can be literal or written numbers. Example: 6, six, 1000, one thousand.
  - non_english - It is reserved for a Question that contain a word written in a Language that is not English. Example: garrafa or banana.

- Your ONLY purpose is to analyze the provided Question and return a JSON list of all categories that applies to that Question.
- A Question can fit into more than one category. If that's the case, return all categories that apply.
- A Question can fit into no categories. If that is the case, return a empty list.
- You must return only the list of categories, nothing else.
- If the Question does not fit into any of the categories, return an empty list.

# Examples of Possible Answers

## Example #1:
Question:
Signer of the Dec. of Indep., framer of the Constitution of Mass., second President of the United States
Answer:
['numbers']

## Example #2: 
Question:
In 1963, live on \"The Art Linkletter Show\", this company served its billionth burger
Answer:
['numbers']

## Example #3:
Question:
A single layer of paper, or to perform one's craft diligently
Answer:
[]

## Example #4:
Question:
In the title of an Aesop fable, this insect shared billing with a grasshopper
Answer:
['non_english']

## Example #5: 
Question:
No. 2: 1912 Olympian; football star at Carlisle Indian School; 6 MLB seasons with the Reds, Giants & Braves
Answer:
['numbers', 'unusual']

## Example #6:
Question:
Around 100 A.D. Tacitus wrote a book on how this art of persuasive speaking had declined since Cicero
Answer:
['unusual', 'non_english', 'numbers']

## Example #7:
Question:
In 1000 Rajaraja I of the Cholas battled to take this Indian Ocean island now known for its tea
Answer:
['unusual', 'non_english', 'numbers']

## Example #8:
Question:
No. 1: Lettered in hoops, football & lacrosse at Syracuse & if you think he couldn't act, ask his 11 "unclean" buddies
Answer:
['non_english', 'numbers']

## Example #9:
Question:
Built in 312 B.C. to link Rome & the South of Italy, it's still in use today
Answer:
['numbers']

## Example #10:
Question:
This housewares store was named for the packaging its merchandise came in & was first displayed on
Answer:
[]

# Question
"{}"
'''

model = 'gemma3:12b'
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama'
)

def generate_datasets():
    '''This function attempts to generate training datasets using LLMs. But the results are not good and should not be used.'''
    unusual_dataset = []
    numbers_dataset = []
    non_english_dataset = []
    max_tags = 10
    is_random_pos = False

    # Opens the questions file
    file_path = os.path.join(os.path.dirname(__file__), '../../files/JEOPARDY_QUESTIONS1.json')
    with open(file_path, 'r') as file:
        # Converts the file data to a python list
        jeopardy_questions = json.load(file)
        # Maximum number of questions available
        size = len(jeopardy_questions)

        # Iterates through the questions
        for i in range(size):
            # Decides if the questions are going to be searched sequentially or randomly
            index = i
            if is_random_pos:
                index = random.randint(0, size - 1)

            # Gets the question
            question = jeopardy_questions[index]
            # Remove & sign...
            question_text = question['question'].replace('&', 'and')
            print(i, question_text)

            # Builds the LLM prompt based on the template
            prompt = prompt_template.format(question_text)

            # Runs the prompt on the LLM
            output = client.completions.create(model=model, prompt=prompt)
            output = output.choices[0].text.strip()
            print(output)

            # Datasets sizes...
            unusual_size = len(unusual_dataset)
            numbers_size = len(numbers_dataset)
            non_english_size = len(non_english_dataset)

            # If was tagged as unusual and still dont have enough questions...
            if output.find('unusual') >= 0 and unusual_size < max_tags:
                # ... append it to the subset.
                unusual_dataset.append(question)
            
            # If was tagged as numbers and still dont have enough questions...
            if output.find('numbers') >= 0 and numbers_size < max_tags:
                # ... append it to the subset.
                numbers_dataset.append(question)
            
            # If was tagged as non_english and still dont have enough questions...
            if output.find('non_english') >= 0 and non_english_size < max_tags:
                # ... append it to the subset.
                non_english_dataset.append(question)
            
            # Datasets sizes...
            unusual_size = len(unusual_dataset)
            numbers_size = len(numbers_dataset)
            non_english_size = len(non_english_dataset)

            # When have enough of each subset, stop looking.
            print('Unusual:', unusual_size, 'Numbers:', numbers_size, 'Non-english:', non_english_size)
            if unusual_size >= max_tags and numbers_size >= max_tags and non_english_size >= max_tags:
                break

    # Saves the subsets to files
    file_path = os.path.join(os.path.dirname(__file__), '../../files/unusual.json')
    with open(file_path, 'w') as file:
        json.dump(unusual_dataset, file, indent=4)

    file_path = os.path.join(os.path.dirname(__file__), '../../files/numbers.json')
    with open(file_path, 'w') as file:
        json.dump(numbers_dataset, file, indent=4)

    file_path = os.path.join(os.path.dirname(__file__), '../../files/non_english.json')
    with open(file_path, 'w') as file:
        json.dump(non_english_dataset, file, indent=4)

if __name__ == '__main__':
    print('Command line run...')
    generate_datasets()

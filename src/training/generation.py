from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama'
)

output = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'Hello!!'
        }
    ],
    model='gemma3:270m'
)

print(output.choices[0].message.content)

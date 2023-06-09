import openai
from setting import OPENAIKEY
openai.api_key=OPENAIKEY
GPT_MODEL ="gpt-3.5-turbo"
query = 'Which athletes won the gold medal in curling at the 2022 Winter Olympics?'

response = openai.ChatCompletion.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about the 2022 Winter Olympics.'},
        {'role': 'user', 'content': query},
    ],
    model=GPT_MODEL,
    temperature=0,
)

print(response['choices'][0]['message']['content'])

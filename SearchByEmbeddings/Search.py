import ast
import pandas as pd
from openai.cli import display
from scipy import spatial
import openai
from setting import OPENAIKEY
openai.api_key=OPENAIKEY
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL ="gpt-3.5-turbo"
embeddings_path = "embeddings/密云区矿区转型绿色数字产业示范园区可行性研究报告.csv"

df = pd.read_csv(embeddings_path)

df['embedding'] = df['embedding'].apply(ast.literal_eval)

def modifyTextDescribe(basecontent:str,requirement:str)->str:
    print(basecontent)
    response = openai.ChatCompletion.create(
        messages=[
            {'role': 'system', 'content': 'You are a good writer.'},
            {'role': 'assistant', 'content': basecontent},
            {'role': 'user', 'content': requirement},
        ],
        model=GPT_MODEL,
        temperature=0,
    )
    print(response)
    return response["choices"][0]["message"]["content"]






# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 3
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

strings, relatednesses = strings_ranked_by_relatedness("浪潮云信息技术股份公司介绍", df, top_n=1)


outstr=strings[0].split("\n\n")[-1]

c=modifyTextDescribe(outstr,"重写为一段企业介绍")

print(c)
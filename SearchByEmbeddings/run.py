import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
datafile_path = "./data/fine_food_reviews_with_embeddings_1k.csv"
from setting import OPENAIKEY

openai.api_key=OPENAIKEY
df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(eval).apply(np.array)

def search_reviews(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "Title")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            #print(r[:200])
            print(r)
    return results


results = search_reviews(df, "delicious soup", n=3)

print(results)
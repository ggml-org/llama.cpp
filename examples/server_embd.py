import asyncio
import asyncio.threads
import requests
import numpy as np


n = 8

result = []

async def requests_post_async(*args, **kwargs):
    return await asyncio.threads.to_thread(requests.post, *args, **kwargs)

async def main():
    model_url = "http://127.0.0.1:6900"
    responses: list[requests.Response] = await asyncio.gather(*[requests_post_async(
        url= f"{model_url}/embedding",
        json= {"content": str(0)*1024}
    ) for i in range(n)])

    for response in responses:
        embedding = response.json()["embedding"]
        print(embedding[-8:])
        result.append(embedding)

asyncio.run(main())

# compute cosine similarity

for i in range(n-1):
    for j in range(i+1, n):
        embedding1 = np.array(result[i])
        embedding2 = np.array(result[j])
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        print(f"Similarity between {i} and {j}: {similarity:.2f}")

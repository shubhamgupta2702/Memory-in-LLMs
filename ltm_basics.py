from langgraph.store.memory import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings

store = InMemoryStore()

namespace = ("user", "u1")

store.put(namespace, "1", {"data": "User likes pizza"})
store.put(namespace, "2", {"data": "User prefers dark mode"})

namespace2 = ("user", "u2")

# Add memories
store.put(namespace2, "1", {"data": "User likes pasta"})
store.put(namespace2, "2", {"data": "User prefers grid style navigation"})

print(store.get(namespace2, "2"))

items = store.search(namespace2)

for item in items:
    print(item.value)
    
    
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

store = InMemoryStore(index={'embed': embeddings, 'dims':1536})

namespace = ('users', 'u1')
store.put(namespace, "1", {"data": "User prefers concise answers over long explanations"})
store.put(namespace, "2", {"data": "User likes examples in Python"})
store.put(namespace, "3", {"data": "User usually works late at night"})
store.put(namespace, "4", {"data": "User prefers dark mode in applications"})
store.put(namespace, "5", {"data": "User is learning machine learning"})
store.put(namespace, "6", {"data": "User dislikes overly theoretical explanations"})
store.put(namespace, "7", {"data": "User prefers step-by-step reasoning"})
store.put(namespace, "8", {"data": "User is based in India"})
store.put(namespace, "9", {"data": "User likes real-world analogies"})
store.put(namespace, "10", {"data": "User prefers bullet points over paragraphs"})

items = store.search(namespace, query="what is the user currently learning", limit=1)

for item in items:
    print(item.value)
    
    
items = store.search(namespace, query="what are user's preferences", limit=3)

for item in items:
    print(item.value)
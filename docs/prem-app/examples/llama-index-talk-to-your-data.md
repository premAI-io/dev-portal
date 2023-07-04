---
sidebar_position: 4
---

# Llama Index: Talk to your Data

- Install the necessary Dependencies

```python
import os

from llama_index.vector_stores import RedisVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index import ListIndex, LLMPredictor, Document

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from llama_index import LangchainEmbedding, ServiceContext

os.environ["OPENAI_API_KEY"] = "random-string"
```

- Create the Documents using Prem Landing Page Content

```python
doc1 = Document(text="Prem is an easy to use open source AI platform. With Prem you can quickly build privacy preserving AI applications.")
doc2 = Document(text="""
Prem App

An intuitive desktop application designed to effortlessly deploy and self-host Open-Source AI models without exposing sensitive data to third-party.

""")
doc3 = Document(text="""
Prem Benefits

Effortless Integration
Seamlessly implement machine learning models with the user-friendly interface of OpenAI's API.

Ready for the Real World
Bypass the complexities of inference optimizations. Prem's got you covered.

Rapid Iterations, Instant Results
Develop, test, and deploy your models in just minutes.

Privacy Above All
Your keys, your models. We ensure end-to-end encryption.

Comprehensive Documentation
Dive into our rich resources and learn how to make the most of Prem.

Preserve Your Anonymity
Make payments with Bitcoin and Cryptocurrency. It's a permissionless infrastructure, designed for you.
""")
```

- Instantiate the LLMs objects accordingly

```python
# Using vicuna-7b-q4
llm_predictor = LLMPredictor(llm=ChatOpenAI(openai_api_base="http://localhost:8111/v1", max_tokens=128))

# Using sentence transformers all-MiniLM-L6-v2
embeddings = OpenAIEmbeddings(openai_api_base="http://localhost:8444/v1")

embed_model = LangchainEmbedding(embeddings)
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm_predictor=llm_predictor)

vector_store = RedisVectorStore(
    index_name="prem_landing",
    index_prefix="llama",
    redis_url="redis://localhost:6379",
    overwrite=True
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = ListIndex.from_documents([doc1, doc2, doc3], storage_context=storage_context)
```

- Query the Index

```python
query_engine = index.as_query_engine(
    retriever_mode="embedding", 
    verbose=True, 
    service_context=service_context
)
response = query_engine.query("What are Prem benefits?")
print(response)
```
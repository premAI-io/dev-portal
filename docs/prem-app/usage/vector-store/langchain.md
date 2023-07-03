---
id: chat-quickstart
title: Quick Start with LangChain
sidebar_label: LangChain
sidebar_position: 2
---

# Quick Start with LangChain

For what concerns Vector Stores Prem doesn't force any inrterface. We only take care of the orchestration. For this reason, you can just run the service and connect to it out of the box. The below example shows how you can use LangChain in order to connect to Redis Vector Store.

### Import the necessary dependencies

```python

!pip install redis

import os

from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores.redis import Redis
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "random-string"

```

### Create some documents that will be indexed

```python
doc1 = Document(page_content="Prem is an easy to use open source AI platform. With Prem you can quickly build provacy preserving AI applications.")
doc2 = Document(page_content="""
Prem App

An intuitive desktop application designed to effortlessly deploy and self-host Open-Source AI models without exposing sensitive data to third-party.

""")
doc3 = Document(page_content="""
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

### Upsert

Instantiate the necessary objects, generate the embeddings and store them into the Vector Store

```python

# Using sentence transformers all-MiniLM-L6-v2
embeddings = OpenAIEmbeddings(openai_api_base="http://localhost:8001/v1")

# Using locally running Redis
url = "redis://localhost:6379"

rds = Redis.from_documents(docs, embeddings, redis_url=url,  index_name="prem_index_test")

query = "What are Prem Benefits?"
docs = vectorstore.similarity_search(query)
print(docs[0].page_content)

```
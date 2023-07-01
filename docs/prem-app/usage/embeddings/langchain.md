---
id: chat-quickstart
title: Quick Start with LangChain
sidebar_label: LangChain
sidebar_position: 2
---

# Quick Start with LangChain

### Import the necessary dependencies

```python
import os

from langchain.embeddings import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "random-string"
```

### Instantiate the Embeddings Object connecting to the service

```python
embeddings = OpenAIEmbeddings(openai_api_base="http://localhost:8000/api/v1")
text = "Prem is an easy to use open source AI platform."
query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])
```

### Generate the Embeddings

```python
embeddings = OpenAIEmbeddings(openai_api_base="http://localhost:8000/api/v1")
text = """

Prem is an easy to use open source AI platform. With Prem you can quickly build provacy preserving AI applications.

Prem App

An intuitive desktop application designed to effortlessly deploy and self-host Open-Source AI models without exposing sensitive data to third-party.

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

"""
query_result = embeddings.embed_query(text)
print(query_result)
```
---
sidebar_position: 3
---

# Langchain: Talk to your Data

- Install the necessary dependencies

```python
import os

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import Qdrant
from langchain.vectorstores.redis import Redis
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "random-string"
```

- Create the Documents using Prem Landing Page Content
    
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

- Using QDrant, Vicuna and Sentence Transformers Running Locally using Prem

```python
# Using vicuna-7b-q4
chat = ChatOpenAI(openai_api_base="http://localhost:8111/v1", max_tokens=128)

# Using sentence transformers all-MiniLM-L6-v2
embeddings = OpenAIEmbeddings(openai_api_base="http://localhost:8444/v1")

# Using locally running Qdrant
url = "http://localhost:6333"

vectorstore = Qdrant.from_documents(
    [doc1, doc2, doc3], 
    embeddings, 
    url=url, 
    collection_name="prem_collection_test",
)

query = "What are Prem Benefits?"
docs = vectorstore.similarity_search(query)
print(docs[0].page_content)
```

- Perform the Query

```python
template = """
You are an AI assistant for answering questions about Prem.
Provide a conversational answer to the question based on the following docouments found using semantic search. Be original, concice, accurate and helpful.

Question: {question}
=========
Context: {context}
=========
Answer in Markdown:
"""  # noqa E501
prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=template,
)
chain = LLMChain(llm=chat, prompt=prompt, verbose=True)

question = "What are Prem Benefits?"
docs = vectorstore.similarity_search(question)
context = docs[0].page_content
chain.run(question=question, context=context)
```

- Example using Redis instead of Qdrant

```python
# Using vicuna-7b-q4
chat = ChatOpenAI(openai_api_base="http://localhost:8001/v1", max_tokens=128)

# Using sentence transformers all-MiniLM-L6-v2
embeddings = OpenAIEmbeddings(openai_api_base="http://localhost:8000/v1")

# Using locally running Redis
url = "redis://localhost:6379"

rds = Redis.from_documents(docs, embeddings, redis_url=url,  index_name="prem_index_test")

query = "What are Prem Benefits?"
docs = vectorstore.similarity_search(query)
print(docs[0].page_content)
```

- Perform the Query

```python
template = """
You are an AI assistant for answering questions about Prem.
Provide a conversational answer to the question based on the following docouments found using semantic search. Be original, concice, accurate and helpful.

Question: {question}
=========
Context: {context}
=========
Answer in Markdown:
"""  # noqa E501
prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=template,
)
chain = LLMChain(llm=chat, prompt=prompt, verbose=True)

question = "What are Prem Benefits?"
docs = vectorstore.similarity_search(question)
context = docs[0].page_content
chain.run(question=question, context=context)
```
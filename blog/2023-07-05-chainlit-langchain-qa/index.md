---
slug: chainlit-langchain-prem
title: Talk to your Data with ChainLit and Langchain
authors: [tiero, filippopedrazzinfp]
tags: [llm, self-hosted, prem, open-source, langchain, chainlit, vicuna-7b, redis, vector-store]
---
<head>
  <meta name="twitter:image" content="./banner.jpg"/>
</head>


Build a chatbot that talks to your data with [Prem](https://premai.io) using `LangChain`, `Chainlit`, `Redis` Vector Store and `Vicuna 7B` model, self-hosted on your MacOS laptop.

![ChainLit x Langchain Screenshot](./chainlit-langchain.gif)


<!--truncate-->

### What is ChainLit?

Chainlit lets you create ChatGPT-like UIs on top of any Python code in minutes!

### What is Langchain?

LangChain is a framework designed to simplify the creation of applications using large language models (LLMs).

### What is Prem?

Prem is a self-hosted AI platform that allows you to test and deploy open-source AI models on your own infrastructure. Prem is open-source and free to use. You can learn more about Prem [here](https://premai.io).


## Talk to your data with Prem

We’re going to build an chatbot QA app. We’ll learn how to:

- Upload a document
- Create vector embeddings from a file
- Create a chatbot app with the ability to display sources used to generate an answer


For this tutorial we are going to use:

- [ChainLit](https://chainlit.io)
- [Langchain](https://docs.langchain.com/docs)
- [Redis](https://redis.io) hosted on the [Prem App](https://premai.io)
- Vicuna 7B model hosted on [Prem App](https://premai.io)

### Step 1: Install Python dependencies

```bash
pip install chainlit langchain redis tiktoken
```

### Step 2: Create a `app.py` file

```bash
touch app.py
```

### Step 3: Add the code!

Please edit accordingly both the Vicuna model and the Redis URL. In this example are respectively `http://localhost:8111/v1` for the Vicuna model and `redis://localhost:6379` for the Redis node.

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQAWithSourcesChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import chainlit as cl

os.environ["OPENAI_API_KEY"] = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=10)

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

`
The answer is foo
SOURCES: xyz
`

Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

@cl.langchain_factory(use_async=True)
async def init():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!", accept=["text/plain"]
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Decode the file
    text = file.content.decode("utf-8")

    # Split the text into chunks
    texts = text_splitter.split_text(text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Redis vector store
    embeddings = OpenAIEmbeddings(
        openai_api_base="http://localhost:8444/v1"
    )
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    # Create a chain that uses the Redis vector store
    chat = ChatOpenAI(
        temperature=0,
        streaming=True,
        max_tokens=128,
        openai_api_base="http://localhost:8111/v1"
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        chat, chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)
    chain.reduce_k_below_max_tokens = True
    chain.max_tokens_limit = 128

    # Save the metadata and texts in the user session
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)

    # Let the user know that the system is ready
    await msg.update(content=f"`{file.name}` processed. You can now ask questions!")

    return chain


@cl.langchain_postprocess
async def process_response(res):
    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = texts[index]
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=source_elements).send()
```

### Step 4: Run the app

```bash
chainlit run app.py
```

You can then upload any `.txt` file to the UI and ask questions about it. If you are using [`state_of_the_union.txt`](https://github.com/hwchase17/langchain/blob/master/docs/extras/modules/state_of_the_union.txt) you can ask questions like `What did the president say about Ketanji Brown Jackson?`


---
sidebar_position: 2
title: Quick Start with LangChain
sidebar_label: LangChain
---

# Quick Start with LangChain

### Import all the necessary dependencies

```python
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

os.environ["OPENAI_API_KEY"] = "random-string"
```

### Instantiate the LLM Object

```python
chat = ChatOpenAI(openai_api_base="http://localhost:8000/v1", max_tokens=128)
```

### Send a message to the LLM

```python
messages = [
    HumanMessage(content="Can you explain what is a large language model?")
]
chat(messages)
```
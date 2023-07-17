---
id: chat-quickstart
title: Quick Start with Python
sidebar_label: Quick Start
sidebar_position: 2
---

# Quick Start with Python

### Install and Import all the necessary dependencies

```python

!pip install requests

import requests

```

### Use requests library to send the http request to the Prem service

```python

prompt = """
Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
But I also have other interests such as playing tic tac toe.
"""

response = requests.post("http://localhost:10111/v1/audio/generation",
                         json={"prompt": prompt})
response_content = requests.get(
    f"http://localhost:10111/files/{response.json()['url']}")

with open("output_file.wav", "wb") as f:
    f.write(response_content.content)

```
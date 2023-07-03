---
id: chat-quickstart
title: Quick Start with OpenAI Client
sidebar_label: Quick Start
sidebar_position: 2
---

# Quick Start with OpenAI Python Client

### Install and Import all the necessary dependencies

```python

!pip install openai
!pip install pillow

import io
import base64
import openai

from PIL import Image

```

### Change the base url in order to point to your Prem services

```python

openai.api_base = "http://localhost:9111/v1"
openai.api_key = "random-string"

```

### Use OpenAI Client in order to generate the images

```python

response = openai.Image.create(
    prompt="Iron man portrait, highly detailed, science fiction landscape, art style by klimt and nixeu and ian sprigger and wlop and krenz cushart",
    n=1,
    size="512x512"
)

image_string = response["data"][0]["b64_json"]

img = Image.open(io.BytesIO(base64.decodebytes(bytes(image_string, "utf-8"))))
img.save("iron_man.jpeg")

```
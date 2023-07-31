---
id: chat-quickstart
title: Quick Start with OpenAI Client
sidebar_label: Quick Start
sidebar_position: 2
---

# Quick Start with Python

### Install and Import all the necessary dependencies

```python

!pip install pillow

import requests
import io
import base64

from PIL import Image

```

### Use Python requests library in order to send requests to the APIs

```python

url = 'http://localhost:8996/v1/images/upscale'
files = {'image': open('iron_man_image.png', 'rb')}  # assuming we have an avg resolution quality iron man image here
data = {
    'prompt': "Super high resolution image of iron man, highly detailed, real life.",
    'n': 1,
    'guidance_scale': 8
}

response = requests.post(url, files=files, data=data)
image_string = response.json()["data"][0]["b64_json"]

img = Image.open(io.BytesIO(base64.decodebytes(bytes(image_string, "utf-8"))))
img.save("iron_man_highres.png", "PNG")

```
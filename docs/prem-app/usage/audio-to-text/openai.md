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

import openai

```

### Change the base url in order to point to your Prem services

```python

openai.api_base = "http://localhost:10111/v1"
openai.api_key = "random-string"

```

### Use OpenAI Client to transcribe the audio file

```python

audio_file = open("./sample.wav", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)
print(transcript)

```
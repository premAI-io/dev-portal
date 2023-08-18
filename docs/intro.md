---
sidebar_position: 1
---

# Introduction

Welcome to Prem! This is the documentation for Prem, the open source platform for running your own AI models on your own servers.

- Be part of the community [joining our Discord](https://discord.com/invite/kpKk6vYVAn).
- To stay in touch [follow us on Twitter](https://twitter.com/premai_io).
- To report bugs or ask for support [open an issue on the Github repository](https://github.com/premAI-io/prem-app).

## What is Prem?

Prem is a platform for running open-source AI models on your own devices, either on MacOS, useful for local development and personal use or on your Linux servers for production. Prem standardize set of **interafces** for AI models, defining types for input and output so that you can run any model and use an unified API to interact with it, regardless of the language and framework it was built with.

Prem is open source and free to use, no data is collected because we CANNOT see your data, as everything runs on-premise.

- Get Started as a Developer
  - [Installation](/docs/category/installation)
  - [Usage](/docs/category/usage)
- Publish your AI Service on Prem
  - [Service Packaging](/docs/category/service-packaging/)

### Main concepts

Prem simplifies the process of running AI models on your own infrastrucutre. It is composed of several components:

- **Prem Service**: An AI model to be served on the platform is called a Prem Service. It is a Docker container that exposes a standardized HTTP API for the **interface** type to interact with the model.
- **Prem Interface**: An interface is the combination of input and output for a Prem Service. It is a semantic type that defines the structure of the data that can be sent to the model and the structure of the data that the model will return. Prem comes with a set of standard interfaces, but you can also create your own. Currently the following interfaces have been created: 😃 Chat, 📕 Embeddings, 🏛️ Vector Store, 🎨 Diffuser. More to come.
- [**Prem App**](/docs/category/prem-app/): the desktop app for MacOS that allows you to run AI models on your own computer.
- [**Prem Daemon**](/docs/category/prem-daemon/): the daemon that runs on your machine and allows you to run AI models on your own infrastructure. It represents the component that exposes all the endpoints necessary to handle the different services and the underlying infrastructure.
- [**Prem Registry**](/docs/registry/): the registry contains all the metadata information in order to run a service in the Prem Ecosystem. Each Prem Service can be published on the public list of AI services.

---
slug: towards-censorship-resistant-architecture
title: Towards a Censorship Resistant Architecture
authors: [filippopedrazzinfp]
tags: [llm, self-hosted, prem, open-source, petals, p2p, decentralized-ai]
description: "Prem - Towards a Censorship Resistant Architecture"
image: "./banner.png"
---

![Prem Banner](./banner.png)

> Horizontal Scaling is all you need.

Prem Labs is introducing **Prem Network**, a decentralized AI Platform enabling consumer devices to run Large Language Models without compromising on quality, aggregating the computational power of devices around the globe. Prem Network is a breakthrough in LLM inference deployment as it enables consumer devices to power GPU-intensive models on the edge. Similar to what BitTorrent did for P2P file sharing, Prem Swarm provides AI inference and training for consumer devices.

## Definitions

- Public Cluster: Cluster composed of GPUs provided by the community. Any Client (Prem App) can access it in order to run the provided models.
- Private Cluster: Cluster composed of GPUs provided by the User. For a more Privacy centric / On-Premise use case.
- Quantization: Model conversion process in order to reduce memory requirements resulting in faster inference.

## A Resource Constrained Environment

Until now, the assumption was clear: consumer hardware cannot leverage state-of-the-art LLMs as they are. A resource-constrained environment forced new paradigms to emerge as an alternative to OpenAI centralization.

Open-source showed how bottom-up innovation can arise from every corner of the world and that big corporations don't have a [moat](https://www.semianalysis.com/i/119223672/we-have-no-moat) for what concerns AI. This has been demonstrated with the advent of new foundational and fine-tuned models like Llama, Llama 2, Falcon 40B, etc.

They are the first step towards creating an alternative to OpenAI centralization.
However, as more powerful open-source models and weights become available to the public, there is still a huge gap between the requirements necessary to run these models and the current computing capabilities of consumer hardware. Today, consumer hardware is still far from leveraging these models' capabilities (e.g., Llama 65B requires around 90GiB of RAM). For this reason, projects like llama.cpp, Alpaca, Vicuna, and GPT4All emerged to run smaller, distilled, quantized versions of these models using consumer devices (CPU and GPU).

On the one hand, this proved that using different methodologies like quantization and fine-tuning smaller models on very clean datasets could reach incredible relative results. On the other hand, these models are still far from ChatGPT and GPT4 quality and semantic understanding.

The low quality is pushing further research to improve these "small" models and increase quality with new methods and techniques to tackle the main issues:

- Context dimension
- Generation quality and hallucination

Some very good papers have come out lately to face these issues (e.g., X, Y), but no major breakthrough has yet emerged. It would seem that perhaps the Transformer model architecture optimization itself is not good enough for the next generation of LLMs with consumer hardware constraints. Only research will be able to answer these open questions.
At Prem, we are focusing on engineering a new paradigm of decentralized and privacy-preserving AI - AI that is always yours. As we push the boundary of what is possible in the deployment of Open Source models at the edge, we are excited to offer an alternative solution to the problem of efficient and optimized deployment of AI models ad the edge.

> As Prem Labs we cannot do bets on research, but only engineering. We can go low level and tackle all the engineering issues step by step solving one problem after the other, but we cannot build a startup based on a research bet.

On the other hand, consumer hardware will be more and more powerful. They are probably able to run bigger and bigger models.

> Similar as above, at Prem Labs we cannot wait that hardware gets better and better of few magnitudes orders before delivering quality to our end users. It's all about balance and bets and even if things are going crazily fast, still ChatGPT needs 175B Parameters to be the model that it is now. Assuming averagely 1GiB per RAM for each 1B Parameters, we are talking about 175GiB of Memory in order to run the state of the art known model. Obviously, even the best consumer device is far from years to be able to have this amount of memory and we really cannot this to happen.

## Twitter vs. Reality

The main problem right now is that we don't have a clear benchmark to use in order to evaluate these models. Both Open LLM Leaderboard and the Alpaca Leaderboard are still far from being able to properly evaluate this new family of LLMs.

A lot of times happened in the last month that a new model reached the top of the Open Source LLM Leaderboard and then after few simple empirical tests you get disappointed by the obvious bad results.
As an industry, we are doing a lot of steps and improvements on a daily basis, but we are not there yet and sometimes the hype confuses what is real and what is not.

## Smaller Models for Smaller Tasks

Another movement that could be considered more like an old way of thinking is to use smaller, specialized models to handle specific tasks. A phenomenon that has not been seen yet in Production so far is to leverage these LLMs for already famous/mainstream NLP tasks like summarization, sentiment analysis, etc. Fine-tuning an LLM and exploiting the semantic understanding of these models to solve atomic NLP tasks, could be another movement that we will see more and more in the future. On the other hand, it's still soon to see these models in Production. A state-of-the-art BERT used for sentiment analysis for example generates responses under 500ms and can be used in a real-time use case. A Llama-7b model 4-bit quantized and fine-tuned on sentiment analysis tasks is still far from this latency requirement even if performances would be probably better in terms of quality.

## A new Paradigm can Quickly Take Over
Even if big improvements are coming at the speed of light, still the necessity to have big models and big hardware is required if you want actual quality. More than ever AI is the era of centralization. Quality models require big GPUs. Only big companies can afford new GPUs able to run these models.

<aside> 💡 Elon Musk ordered 10000 GPUs and he took two months in order to finalize the order. </aside>

Given the current limitations, a torrent-based inference engine seems the only solution in order to deliver quality models to the end consumer. On the other hand, the first working prototype came out with petals. The underlined algorithm was theoretically implemented a few years ago by X. Still some limitations are present and a lot can be done in terms of optimization improvements for what concerns the distribution algorithm.

## What are the Benefits of this New Paradigm?

- 🚀 Run Prem in your Infrastructure in one click: Install Prem on your Mac or Linux server and start using the Prem Services.
- 🔑 Verifiable Privacy: Never expose your sensitive data with any third parties. We ensure end-to-end encryption.
- 🛠️ Seamless Integration and Unified Environment. Simple, easy to use APIs.

### Additionally:

- 🖥️ No need for Big GPUs. For both Enterprises, Small Businesses, Startups, and Research teams: most companies cannot afford current GPU costs.
- 🔑 Uncensored Models. Censorship Resistant AI Models.
- 🤦‍♂️ ChatGPT Comparable Quality in consumer ends. Consumers can now leverage big models on their laptops without compromising privacy.

Enterprises will still benefit from this new paradigm. They will be able to use the Public Cluster for what concerns high-load operation (peaks and fine-tuning), but also create their own Private Cluster using older GPUs. All the previous benefits will apply but with the additional point of being able to use older generations of GPUs which is the standard case in an On-Premise infrastructure removing the need for expensive and difficult-to-find GPUs.

On the other hand, consumers will be able to offer their personal devices as atomic resources for the Public Cluster and receive a monetary incentive for the computation (Fiat, Bitcoin or Credits).

## B2B Magic

While companies (our potential clients) they just care about privacy, DX and costs. As soon as we show them what we can do they will buy, but they don't care about underlying technology and innovation. They don't care if we use torrent or something else. They will just see MAGIC. Big models running on their shitty GPUs
And with Prem we fill the gap. You need 150 GiB to run Bloom 170B and you only have 100GiB total with your shitty GPUs… then easy. Run the hidden layers on Prem GPUs

## Privacy preserving by design

## Anyone Can Contribute

I am a Developer and I have a Mac M1. I spend 8/10 hours per day at the computer and I usually run 4/5 docker containers and multiple desktop apps as VSCode, Firefox, Notion for my day to day operations. The rest of the time, which is around 14 hours / day, my Mac is idle. I have the best consumer hardware on earth and None can exploit it. At Prem Labs We believe in a world where anybody can contribute to the Public Cluster and receive monetary incentives for that.

## What About Other Devices?
Apart from having a Mac M1, I also have an iPhone 14 that I basically use 2 hours per day and another Mac Book Pro from 2018 that is 100% idle. Running Prem App on both my iPhone and my second Mac it's just a matter of installing the respective native Apps and Done. I am part of the Public Cluster.

## Conclusion

> We are the AI Rebels.
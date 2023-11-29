---
slug: petals-private-swarm
title: How to Deploy your Petals Private Swarm
authors: [filopedraz]
tags: [llms, petals, swarm, private, prem]
description: "Get your own Petals Private Swarm up and running in 5 minutes"
---

<!--truncate-->

## Introduction to Petals

### What is Petals?

### Some Definitions and Concepts

- **DHT**: Distributed Hash Table
- **Torrent**: A torrent is a file sent via the BitTorrent protocol. It can be just about any type of file, such as a movie, song, game, or application.
- **Swarm**: A swarm is a group of peers that share a torrent and are both uploading and downloading the torrent's content.
- **Peer**: A peer is one instance of a BitTorrent client running on a computer on the Internet to which other clients connect and transfer data.
- **Tracker**: A tracker is a server that keeps track of which seeds and peers are in the swarm.
- **Seed**: A seed is a client that has a complete copy of the data of a certain torrent. Once your BitTorrent client finishes downloading, it will remain open until you click the Finish button (or otherwise close it). This is known as being a seed or seeding.

### How does Petals work?

## Deploy your Petals Private Swarm

### 1. Run a Backbone Server

Create a DO Machine with at least 2GiB of RAM and 1 CPU and Install all the necessary dependencies (Docker).

```bash
docker run -d --net host --ipc host --volume petals-cache-backbone:/cache --name backbone --rm learningathome/petals:main python -m petals.cli.run_dht --host_maddrs /ip4/0.0.0.0/tcp/31337 --identity_path bootstrap1.id
```

Check the logs of the `backbone` containers in order to get the IPs.

### 2. Contribute to the Swarm

#### Cloud GPU Instance (NVIDIA)

Create a Machine on `Paperspace` or `DataCrunch` and run the following command in order to join the Swarm

```bash
docker run -d --net host --ipc host --gpus all --volume petals-cache-node1:/cache --name node1 --rm learningathome/petals:main python -m petals.cli.run_server --port 31330 --num_blocks 20 petals-team/StableBeluga2 --initial_peers /ip4/209.38.217.30/tcp/31337/p2p/QmecL18cmRaDdAcRmA7Ctj1gyAeUYG433WppA1UWTHTew6 /ip4/127.0.0.1/tcp/31337/p2p/QmecL18cmRaDdAcRmA7Ctj1gyAeUYG433WppA1UWTHTew6
```

Where the `--initial_peers` variable should be filled with the logs you get from the `backbone` peer.

#### Mac

```bash
brew install python
python3 -m pip install git+https://github.com/bigscience-workshop/petals
python3 -m petals.cli.run_server --public_name prem-app petals-team/StableBeluga2 --initial_peers /ip4/209.38.217.30/tcp/31337/p2p/QmecL18cmRaDdAcRmA7Ctj1gyAeUYG433WppA1UWTHTew6 /ip4/127.0.0.1/tcp/31337/p2p/QmecL18cmRaDdAcRmA7Ctj1gyAeUYG433WppA1UWTHTew6
```

#### Prem

- Install Prem Desktop App
- Go to `Settings`
- Enable `Swarm Mode`

### 3. Monitor your Private Swarm

- Run Prem Explorer

### 4. Consume your Private Swarm

- Run Prem App and connect to your Private Swarm

## Conclusion
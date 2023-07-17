---
id: install-linux-production
title: Install on Linux for Production
sidebar_label: Install on Linux for Production
sidebar_position: 2
---

# Linux for Production

Install everything needed to run Prem on Ubuntu/Debian server

```bash
wget -q https://get.prem.ninja/install.sh -O install.sh; sudo bash ./install.sh
```

## Uninstall Prem 

If you want to uninstall Prem, you can run the following commands

```bash
# navigate to prem directory
cd ~/prem/

# stop docker-compose services
docker-compose down
```
------

If you encounter issues or you want to build the Prem App docker image inside your Linux server

### CPU 
```bash
git clone https://github.com/premAI-io/prem-app.git
cd ./prem-app
docker-compose up -d
```

### GPU (NVIDIA)
```bash
git clone https://github.com/premAI-io/prem-app.git
cd ./prem-app
docker-compose up -f docker-compose.yml -f docker-compose.gpu.yml -d
```

And you will have the UI at `http://{localhost|server_ip}:1420`. 

> Make sure that in `Settings` the Backend URL is set to `http://{localhost|server_ip}:8000`.
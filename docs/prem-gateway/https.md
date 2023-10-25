---
id: https
title: Automatic HTTPS
sidebar_label: Automatic HTTPS
sidebar_position: 2
---

At start, the Prem App is accessible via IP that is printed by the installation script. This IP is Prem Gateway IP address.

## How it works?

Once you setup [your own domain](./domains.md), Prem App will automatically generate SSL certificate for your domain and enable HTTPS for the application. Nothing to do here.

Prem Services are accessible on subdomain level, for example, if you have Chat model named `llama`, it will be accessible on `https://llama.example.com/v1/chat/completions` endpoint
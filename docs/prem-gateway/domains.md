---
id: domains
title: Domains
sidebar_label: Domains
sidebar_position: 1
---

At start, the Prem App is accessible via IP that is printed by the installation script. This IP is Prem Gateway IP address.

## How it works?

Domains feature gives you option to easily map your domain to your Prem App application.
It is necessary to insert two A records:
- A `example.com` to `<IP_GATEWAY>`
- A `*.example.com` `<IP_GATEWAY>`
You can also use a subdomain ie. `prem.example.com` and `*.prem.example.com`

After this is done you need to add your domain in **Prem App > Settings > Domains**, and you are ready to go.

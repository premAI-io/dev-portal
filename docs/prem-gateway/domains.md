---
id: domains
title: Domains
sidebar_label: Domains
sidebar_position: 1
---

At start, the Prem App is accessible via IP that is printed by the installation script. This IP is Prem Gateway IP address.

## How it works?

Domains feature gives you option to easily map your domain to your Prem App application.
It is necessary to insert A record for your subdomain(with wildcard) and point it to Prem Gateway IP address.
For example, considering your domain is `example.com`, you need to insert A record for `*.example.com` and point it to Prem Gateway IP address.

After this is done you need to add your domain in **Prem App > Settings > Domains**, and you are ready to go.

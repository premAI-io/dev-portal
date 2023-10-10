---
id: usage
title: Usage
sidebar_label: Usage
sidebar_position: 2
---

## Dns
Dns feature gives you option to easily map your domain to your Prem App application. <br/>
Initially Prem App is accessible via IP that is printed by installation script. This IP is Prem Gateway IP address. <br/>
It is necessary to insert A record for your subdomain(with wildcard) and point it to Prem Gateway IP address. <br/>
For example, considering your domain is `example.com`, you need to insert A record for `*.example.com` and point it to Prem Gateway IP address. <br/>
After this is done you need to add your domain in Prem App' settings page, and you are ready to go. <br/>
Instead of IP address, you can use your domain to access Prem App. <br/>
Prem-Services are accessible on subdomain level, for example, if you have service named `service1`, it will be accessible on `service1.example.com`. <br/>
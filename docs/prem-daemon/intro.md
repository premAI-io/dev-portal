---
sidebar_position: 1
---

# Overview

Prem Daemon represents the component that handle the orchestraction of the Prem Services. It is responsible to launch the requested service and to return the response to Prem App. You can check the repository [here](https://github.com/premAI-io/prem-daemon).

The information flow works as following:

Prem App sends HTTP requests to Prem Daemon. Prem Daemon is responsible to launch the requested service and to return the response to Prem App. Prem Daemon using Docker SDK starts the requested service as a Docker container. Based on the interface exposed by the service, Prem App can directly interact with the service or it can use Prem Daemon as a proxy.

### Registry

When running Prem Daemon, you can specify an environment variable `PREM_REGISTRY_URL`. This variable is the URL of the registry to use. The registry contains all the metadata information in order to run a service in the Prem Ecosystem. Each Prem Service can be published on the public list of AI services. By default, Prem Daemon uses the latest stable version of Prem Registry: https://raw.githubusercontent.com/premAI-io/prem-registry/main/manifests.json.

More information about Prem Registry and how to publish your service can be found [here](/docs/registry/).
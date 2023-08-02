# 🤖 Prem Developer Portal

Source code for <https://dev.premai.io>.

## Contributing

You can open this repository in a [Dev Container](https://containers.dev), or alternatively follow the instructions below.

### Run the Web Server locally

Requires [yarn](https://yarnpkg.com/getting-started/install).

```sh
yarn install  # install the necessary dependencies
yarn start    # run the webserver
```

### Generate OpenAPI Documentation

The [docusaurus-openapi-docs](https://github.com/PaloAltoNetworks/docusaurus-openapi-docs) plugin is used to include the `services` OpenAPI documentation. Run this before `yarn start`:

```sh
yarn docusaurus gen-api-docs all
```

[New specifications can be added](https://github.com/PaloAltoNetworks/docusaurus-openapi-docs#configuring-docusaurusconfigjs-plugin-and-theme-usage) and the docs regenerated using `yarn docusaurus gen-api-docs <id>`.

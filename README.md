# 🤖 Prem Developer Portal

## Contributing

### Run the Web Server locally

```bash
# install the necessary dependencies
yarn install 

# run the webserver
yarn start
```

### Generate OpenAPI Documentation

In order to include the services OpenAPI documentation in the Developer Portal, we leverage [docusaurus-openapi-docs](https://github.com/PaloAltoNetworks/docusaurus-openapi-docs) plugin. Follow the documentation [here](https://github.com/PaloAltoNetworks/docusaurus-openapi-docs#configuring-docusaurusconfigjs-plugin-and-theme-usage) in order to add new specifications and then generate the doc using the following command:

```bash
yarn docusaurus gen-api-docs <id>
```
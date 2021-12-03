# Continuous Integration (CI) & Continuous Deployment (CD)

CI and CD can be performed using any platform like `Azure DevOps Pipeline` or `GitHub Actions`, etc. where the following `make` commands in [Makefile](../../Makefile) might be useful.

- CI: execute `make ci` from the Pipeline/Action stage.
- CD: execute `make cd` from the Pipeline/Action stage.

**NOTE:** Set env variables - `DATABRICKS_HOST`, `DATABRICKS_TOKEN` in the environment prior executing CD stage.

## Reference

- [Design a CI/CD pipeline using Azure DevOps](https://docs.microsoft.com/en-us/azure/architecture/example-scenario/apps/devops-dotnet-webapp)
- [GitHub Actions](https://docs.github.com/en/actions)
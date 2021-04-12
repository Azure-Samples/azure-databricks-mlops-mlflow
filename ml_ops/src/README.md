# SRC

## Overview

Source code for MLOps, based on  -

1. `diabetes_mlops` contains MLOps source code for `diabetes` machine learning code.
2. The MLOps Python functions will be called from orchestrator Databricks Notebook.
3. Ops related integrations (MLflow and Application Insights Metrics, Tracing, etc.) may happen in MLOps source code.
4. Mostly no Machine Learning (data science) related logics will be written in MLOps.
5. DataFrame I/O will happen in orchestrator Databricks Notebook, not in MLOps source code.

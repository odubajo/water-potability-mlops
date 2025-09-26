# https://dagshub.com/odubajo/water-potability-mlops.mlflow

import dagshub
dagshub.init(repo_owner='odubajo', repo_name='water-potability-mlops', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
# from datetime import datetime
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import pickle
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule, CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    log=get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    log.info(f"MSE is {mse}")
    mlflow.log_metric('MSE', mse)
    return

@task
def get_paths(date):
    if date==None:
        train_date=datetime.datetime.now()
        val_date=datetime.datetime.now()

    else:
        date=datetime.datetime.strptime(date, '%Y-%m-%d')
        train_date = date
        val_date = date

    train_date = train_date - relativedelta(months=2)
    val_date = val_date - relativedelta(months=1)
    train_date = train_date.strftime('%Y-%m')
    val_date = val_date.strftime('%Y-%m')
    train_path = "./data/fhv_tripdata_" + train_date + ".parquet"
    val_path = "./data/fhv_tripdata_" + val_date + ".parquet"

    return train_path,val_path


@flow(task_runner=SequentialTaskRunner())
def main(date="2021-08-15"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Week-3 Experiment with Prefect")

    with mlflow.start_run():
        train_path, val_path= get_paths(date).result()

        categorical = ['PUlocationID', 'DOlocationID']

        df_train = read_data(train_path)
        df_train_processed = prepare_features(df_train, categorical)

        df_val = read_data(val_path)
        df_val_processed = prepare_features(df_val, categorical, False)

        # train the model
        lr, dv = train_model(df_train_processed, categorical).result()
        run_model(df_val_processed, categorical, dv, lr)

        model_name = f"model-{date}.bin"
        with open(model_name,'wb') as f:
            pickle.dump(lr,f)

        dv_name = f"dv-{date}.bin"
        with open(dv_name,'wb') as f:
            pickle.dump(dv,f)

main()

DeploymentSpec(
    name="model_training",
    flow=main,
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"
    ),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)



from airflow import DAG
from airflow.operators.bash import BashOperator

from datetime import datetime, timedelta


PROJECT_DIR = "/opt/airflow/project"


default_args = {

    "retries": 2,

    "retry_delay": timedelta(minutes=2)

}



with DAG(

    dag_id="pricing_pipeline",

    description="Dynamic pricing ML pipeline",

    start_date=datetime(2026,1,1),

    schedule_interval="@daily",

    catchup=False,

    default_args=default_args,

    tags=["pricing"]

) as dag:

    # NEW: folds today's streamed data (accumulated by stream_processor.py
    # into streaming_features_log.csv) into master_features.csv BEFORE
    # retraining runs, so retrain_models actually trains on new data
    # instead of the exact same historical file every day.
    update_master_data = BashOperator(

        task_id="update_master_data",

        bash_command=f"""

        cd {PROJECT_DIR}

        python src/ml/update_master_features.py

        """

    )

    retrain_models = BashOperator(

        task_id="retrain_models",

        bash_command=f"""

        cd {PROJECT_DIR}

        python src/ml/retrain_all.py

        """

    )



    run_batch_optimization = BashOperator(

        task_id="run_batch_optimization",

        bash_command=f"""

        cd {PROJECT_DIR}

        python src/ml/optimization.py

        """

    )


    update_master_data >> retrain_models >> run_batch_optimization
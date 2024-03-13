import os
import json
import pandas as pd
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional
import subprocess
from jinja2 import Environment, FileSystemLoader, DebugUndefined

from src.config import (
    DATA_DIR,
    QUERY_FOLDER,
    DBT_FOLDER,
    load_query_params,
)
from loguru import logger
from src.utils import timehop
from src.dbconnector import redshift, get_mysql_connection
from src.model import Models

class DatasetType(Enum):
    """Enums over types of datasets

    Args:
        Enum (str): Type of dataset
    """

    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"
    INFERENCE = "inference"


class DBTrunner:
    """Class responsible to run DBT job with runtime vars
    to support building of dataset in Redshift.
    """

    def __init__(self, dbt_project_path: str) -> None:
        self.dbt_project_path = dbt_project_path

    def run(
        self,
        vars: Optional[str] = {},
        model: Optional[str] = "dataset__seller_fraud_model",
        threads=6,
    ) -> None:
        """Runs the DBT model with the given variables and project dir"""
        dbt_vars = json.dumps(vars)
        profiles_path = os.path.join(self.dbt_project_path, "profiles")

        dbt_command = f"""dbt run 
        --profiles-dir {profiles_path}
        --project-dir {self.dbt_project_path} 
        --models {model} 
        --threads {threads}
        --vars"""
        dbt_command_args = dbt_command.split()
        dbt_command_args.append(f"{dbt_vars}")
        subprocess.run(dbt_command_args)

    def get_data(
        self,
        schema: Optional[str] = "seller_fraud_model_feature_store",
        table: Optional[str] = "dataset__seller_fraud_model",
    ) -> pd.DataFrame:
        """Returns a dataframe of the table created by dbt

        Returns:
            pd.DataFrame: DataFrame from running DBT
        """

        query = f"""SELECT * FROM {schema}.{table}"""
        df = redshift(query=query)
        return df


class ProcessSQL:
    """
    This class is responsible for defining method
    to render SQL queries to desired destination
    redshift, or mysql instances.
    """

    ENV_CACHE = {}

    def __init__(
        self, node: str, queries_path: str, template_variables: dict = {}
    ) -> None:
        self.queries_path = queries_path
        self.template_variables = template_variables
        self.node = node
        self.queries = {}
        self.mysql_connection_map = load_query_params().get("connection_mapping")

        if self.ENV_CACHE.get(self.queries_path):
            self.env = self.ENV_CACHE[self.queries_path]
        else:
            self.env = Environment(
                loader=FileSystemLoader(self.queries_path),
                undefined=DebugUndefined,
                cache_size=40,
            )

    def __process_variables(self) -> None:
        # loading query params
        query_params = load_query_params().get("variables", {})
        variables_per_node = query_params.get("nodes", {})

        # removing node specific variables
        # query_params.pop('nodes')

        # getting the variables for the node
        variables = variables_per_node.get(self.node)

        # updating the non node specific vars to node vars
        variables.update(query_params)
        variables.update({"node": self.node})

        # updating the vars with values passed on runtime
        variables.update(self.template_variables)
        self.template_variables = variables

    def __make_queries(self, filter_f: Optional[Callable] = None) -> None:
        filter_f = (
            (lambda x: f"combined{os.sep}" not in x) if not filter_f else filter_f
        )
        for template in self.env.list_templates(filter_func=filter_f):
            query = Path(template).stem
            self.queries[query] = self.env.get_template(template).render(
                **self.template_variables
            )

    def get_queries(self, filter_for_sql_files: Optional[Callable]) -> Dict[str, str]:
        """Get querries from the sepecified query folder and renders the Jinja Template
        with the passed varaibles to the class.
        Uses __process_varaibles and __make_queries

        Args:
            filter_for_sql_files (Optional[Callable]): A function to pass which filters sql template files
            from the query list.

        Returns:
            Dict[str,str]: Dict of filename as key and rendered queries as value
        """
        self.__process_variables()
        self.__make_queries(filter_f=filter_for_sql_files)
        return self.queries

    def __get_data_from_queries(
        self, filter_for_sql_files: Optional[Callable]
    ) -> Dict[str, pd.DataFrame]:
        query_data = {}
        processed_queries = self.get_queries(filter_for_sql_files=filter_for_sql_files)
        for query_name, query in processed_queries.items():
            try:
                if self.node == "mysql":
                    node = get_mysql_connection(
                        node=self.mysql_connection_map.get(query_name)
                    )
                else:
                    node = redshift
                query_data[query_name] = node(query=query)
            except Exception as e:
                logger.error(f"Failure to load query {query}: {e}")
        return query_data

    def get_data(self) -> pd.DataFrame:
        """Gets data from queries and joins it based on the logic defined.
        This function is not abstract enough.
        Returns:
            pd.DataFrame: Joined dataframe to create the training data
        """
        df_from_queries = self.__get_data_from_queries(filter_for_sql_files=None)

        return df_from_queries

    def get_data_combined(self) -> pd.DataFrame:
        """Gets data by combining the user based query into one and then join it
        with payouts data.
        This function is not abstract enough.

        Returns:
            pd.DataFrame: Joined dataframe to create the training data
        """
        filter_func = lambda x: f"combined{os.sep}" in x
        df_from_queries = self.__get_data_from_queries(filter_for_sql_files=filter_func)

        df = df_from_queries["website_data_combined"]
        # remove duplicated columns NOT duplicates

        df = df.loc[:, ~df.columns.duplicated()]
        df = df.merge(df_from_queries["payoutmethod"], how="left", on="user_id")
        return df

class Dataset:
    """Class responsible for generating dataset from
    Redshift or production database. This class uses DBT and templated
    SQL to issue SQL query to any of the underlying data source.
    """

    ALLOWED_SOURCES_FOR_DATASET = ["prod", "redshift"]

    def __init__(
        self,
        experiment_model: Optional[Models] = None,
        train_start_date: Optional[str] = timehop(-365),
        train_end_date: Optional[str] = timehop(0),
        save_experiment_data: Optional[bool] = True,
        load_from_cache: Optional[bool] = False,
    ) -> None:
        self.experiment_model = experiment_model.value() if experiment_model else None
        self.save_experiment_data = save_experiment_data
        self.load_cached_data = load_from_cache
        self.queries_path = QUERY_FOLDER
        self.external_data_dir = os.path.join(DATA_DIR, "external")
        self.raw_data_dir = os.path.join(DATA_DIR, "raw")
        self.transformed_data_dir = os.path.join(DATA_DIR, "processed")
        self.raw_data_location = os.path.join(
            self.raw_data_dir, f"seller_fraud_features_raw.csv"
        )
        self.transformed_data_location = os.path.join(
            self.transformed_data_dir, f"seller_fraud_features_transformed.csv"
        )

        self.render_vars = {
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
        }


    def make_training_data(
        self,
        source: Optional[str] = "redshift",
        queries_path: Optional[str] = QUERY_FOLDER,
        dbt_project: Optional[str] = DBT_FOLDER,
    ) -> pd.DataFrame:
        """This method exposes funtionality to create dataset from
            redshift or prod db using DBT or SQL template files

        Args:
            source (Optional[str], optional): Source of data. Defaults to "redshift".
            queries_path (Optional[str], optional): Location Query Path. Defaults to QUERY_FOLDER.
            dbt_project (Optional[str], optional): Location to DBT Project. Defaults to DBT_FOLDER.

        Returns:
            pd.DataFrame: Train data obtained from data source.
        """
        queries_path = queries_path if queries_path else self.queries_path
        if source in self.ALLOWED_SOURCES_FOR_DATASET:
            if self.load_cached_data and os.path.exists(self.transformed_data_location):
                transformed_data = pd.read_csv(self.transformed_data_location)
                return transformed_data

            elif self.load_cached_data and os.path.exists(self.raw_data_location):
                raw_data = pd.read_csv(self.raw_data_location)
                transformed_data = self.__transform_raw_data(raw_df=raw_data)
                return transformed_data

            else:
                if source == "mysql":
                    raw_data = self.__get_raw_dataset_from_prod(
                        queries_path=queries_path
                    )

        else:
            raise ValueError
        return transformed_data


if __name__ == "__main__":
    print('hello world!')
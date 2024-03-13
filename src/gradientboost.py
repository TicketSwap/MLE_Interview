import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Optional, Tuple

from src import LOG_LEVEL
from sklearn.base import BaseEstimator
from category_encoders import LeaveOneOutEncoder
from sklearn.ensemble import GradientBoostingClassifier

from src.features import Features, features
from src.model.base import BaseModel

from loguru import logger
from src.utils import classifier_analysis

class GradientBoostModel(BaseModel):
    """
    This class is generates a gradient boost classifier model
    based on the generic template mentioned in the BaseModel
    """
    #### Update the model hyperparameters here ####
    DEFAULT_PARAMS: dict[str, Any] = {
        "loss": "deviance",
        "n_estimators": 3000,
        "max_depth": 3,  # 6
        "learning_rate": 0.01,
        "subsample": 0.5,
        "max_features": "auto",
        "n_iter_no_change": 50,
        "tol": 0.001,
        "verbose": 1,
        "random_state": 42,
    }
    #### -------------------------------------- ####

    def __init__(
        self,
        datapreparer: Optional[BaseEstimator] = LeaveOneOutEncoder,
        model_params: Optional[dict] = None,
        features: Optional[Features] = features,
        context: Optional[dict] = None,
    ) -> None:

        self.model_params = model_params if model_params else self.DEFAULT_PARAMS
        self.label = features.get_label()
        self.predictor = GradientBoostingClassifier(**self.model_params)
        self.datapreparer = datapreparer
        self.features = features
        self.trained = False
        self.model_context = context

    @property
    def isTrained(self) -> bool:
        return self.trained

    def save(self, path: Optional[str] = None):
        """method to save the model to local filesystem"""
        # use the path provided or use the default path
        local_save_path = path if path else self.model_local_path

        # create the directory if it doesn't exist
        if not os.path.exists(local_save_path):
            os.makedirs(local_save_path)

        # save the model artifacts
        for artifact in self.model_artifacts.keys():
            save_location = Path(local_save_path) / f"{artifact}.pkl"
            logger.debug(f"Saving {artifact} to {save_location}")
            joblib.dump(self.model_artifacts[artifact], save_location)

    @staticmethod
    def train_test_split(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Method to generate train and test datafame from a given dataset

        Args:
            df (pd.DataFrame): Dataset to train on.
            by_user (bool, optional): split dataframe by user. Defaults to True.
            eval_set_percentage (int, optional): test set percentage to get from total dataset. Defaults to 0.1.
            label (Optional[str], optional): Label in dataframe to be consider as target column. Defaults to None.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframe
        """
        # create the test=
        df_train = df.sample(frac=0.8, random_state=42)
        de_eval = df.sample(frac=0.3, random_state=42)
        logger.info(
            f"Train and test dataframes generated with {df_train.__len__()} and {de_eval.__len__()} samples respectively."
        )
        return df_train, de_eval

    @staticmethod
    def calculate_sample_weight(y: pd.DataFrame) -> pd.DataFrame:
        """Gives more weight to fraud records;
        So the overall weight for all fraud records is balanced
        with the overall weight for the non fraud records

        Args:
            y (pd.DataFrame): A label dataframe to calculate the sample weight from

        Returns:
            pd.DataFrame: dataframe with weight for each element in y
        """
        multiplier = int((1 - y.mean()) / y.mean())
        return y.apply(lambda ac: multiplier if ac > 0 else 1)

    def train(
        self, data: pd.DataFrame, test_set_percentage: Optional[int] = 0.1
    ) -> float:
        """Method for starting training of the model.

        Args:
            data (pd.DataFrame): Train dataset
            test_set_percentage (Optional[int], optional): Test dataset % taken from data . Defaults to 0.1.
                                                        Uses staticmethod train_test_split to generate dataset.

        Returns:
            float: Accuracy of the model
        """

        df_train, df_test = self.train_test_split(
            data, eval_set_percentage=test_set_percentage
        )

        x_train, y_train = self.feature_target_split(
            data=df_train,
            feature_names=self.features.get_all(),
            label_name=self.features.get_label(),
        )

        sample_weight = self.calculate_sample_weight(y_train)

        self.datapreparer = self.datapreparer(
            cols=features.get_categorical(), sigma=0.5
        )

        self.datapreparer.fit(x_train, y_train)

        x_train = self.datapreparer.transform(x_train).fillna(0)
        self.predictor.fit(x_train, y_train, sample_weight=sample_weight)

        x_test, y_test = self.feature_target_split(
            data=df_test,
            feature_names=self.features.get_all(),
            label_name=self.features.get_label(),
        )
        self.eval(x_test=x_train, y_test=y_train, dataset_name="Train")

        data = self.eval(x_test=x_test, y_test=y_test, dataset_name="Test")

        return data.attrs["score"]

    def predict(self, x: pd.DataFrame) -> dict:
        """Gives the confidence score of the models fraud prediction.

        Args:
            x (pd.DataFrame): Data point from which prediction is to be made.

        Returns:
            dict: The confidence score for x to be fraud or not.
        """

        x = x[self.features.get_all()]
        x = self.datapreparer.transform(X=x).fillna(0)
        prediction = self.predictor.predict(x)
        confidence = self.predictor.predict_proba(x)[:, 1]
        return {"score": confidence}

    def eval(
        self, x_test: pd.DataFrame, y_test: pd.DataFrame, dataset_name: str = "Test"
    ) -> np.array:
        """Evaluates model performance against a test set.

        Args:
            x_test (pd.DataFrame): Features of the eval cases
            y_test (pd.DataFrame): Label value of eval set
            dataset_name (str, optional): Type of eval. Defaults to "Test".

        Returns:
            np.array: array of prediction probabilities
            Note: model accuracy can be found from the dataframe attributes. e.g. data.attrs["score"]
        """
        x_test: pd.DataFrame = x_test[self.features.get_all()]
        x_test: pd.DataFrame = self.datapreparer.transform(x_test).fillna(0)
        y_test_predict = self.predictor.predict(x_test)
        y_test_predict_proba = self.predictor.predict_proba(x_test)[:, 1]

        logger.info(f"% Frauds in dataset {dataset_name} {y_test.mean():.1%}")

        threshold: float = 0.15

        # Calculates model metrics for threshold.
        # Generates a list of
        percent_of_listings = (y_test_predict_proba >= threshold).mean()
        ds_name: str = f"{dataset_name} (@ threshold = {threshold}; {percent_of_listings:.2%} of listings)"
        score = classifier_analysis(
            y_test, y_test_predict_proba >= threshold, dataset_name=ds_name
        )

        data: pd.DataFrame = pd.DataFrame(
            {"labels": y_test, "predictions": y_test_predict_proba}
        )
        data.attrs["score"] = score
        return data
    
    def model_type(self):
        return "Gradient_Boost"



if __name__ == "__main__":

    predictor_parameters = {
        "loss": "deviance",
        "n_estimators": 50,
        "max_depth": 3,  # 6
        "learning_rate": 0.04,
        "subsample": 0.2,
        "max_features": "auto",
        "n_iter_no_change": 50,
        "tol": 0.001,
        "verbose": 1,
        "random_state": 42,
    }
    m = GradientBoostModel(model_params=predictor_parameters)

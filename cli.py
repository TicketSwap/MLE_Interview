import click
import pandas as pd
from src.etl import Dataset
from src.gradientboost import GradientBoostModel

pd.set_option("display.max_rows", 100)

@click.command()
@click.option(
    "--mode",
    type=click.Choice(["train"]),
    default="train",
)

def cli(mode):
    if mode == "train":
        train()
    else:
        pass

def train(
    data: Dataset = None,
    hparams: dict = None,
    save_model: bool = True,
) -> float:
    """
    Instantiates a Model object, loads the training data, splits the data into
    training and test, stores the files locally and in s3 and finally trains the model.

    Returns
    -------
    model_score [float]: The accuracy score of the model [0-1]
    """
    model: GradientBoostModel = GradientBoostModel

    data = Dataset(
        experiment_model=model.model_type(),
        load_from_cache=False,
        save_experiment_data=True,
    ).make_training_data()

    if hparams is not None:
        for key, value in hparams.items():
            model.model_params[key] = value

    model_score = model.train(data=data, test_set_percentage=0.1)

    return {"model_score": model_score, "model": model}

if __name__ == "__main__":
    cli()

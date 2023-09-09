"""
Task to train the models with the best training parameter I have found so far.
"""

import pytask
import os
from train_liar_model.main_functions import train_and_evaluate


@pytask.mark.depends_on(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..",
                "models",
                "random_forest"
            )
        )
)
def task_train_random_forest_model(depends_on):
    """
    Train a model with Bert algorithm and a model with random forest algorithm

    Args:
    - depends_on (path): Output path for the model and log file

    Returns:
    - None
    """

    # Parameter for the training of a model with random forest algorithm
    random_forest_parameter_normal = {
        "n_estimators": 450,
        "max_depth": 80,
        "min_samples_leaf": 2,
        "min_samples_split": 30
    }

    # Model trained with random forest algorithm
    rf_model_path = os.path.abspath(os.path.join(
        depends_on,
        "random_forest_model.pkl"
        ))
    rf_log_path = os.path.abspath(os.path.join(
        depends_on,
        "random_forest_training.log"
        ))
    train_and_evaluate(
        algorithm="random_forest",
        train_type="basic",
        log_path=rf_log_path,
        save_path=rf_model_path,
        max_features=5000,
        training_parameters=random_forest_parameter_normal
    )


@pytask.mark.depends_on(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..",
                "models",
                "bert"
            )
        )
)
def task_train_models(depends_on):
    """
    Train a model with Bert algorithm and a model with random forest algorithm

    Args:
    - depends_on (path): Output path for the model and log files

    Returns:
    - None
    """

    # Parameter for the training of a model with BERT algorithm
    bert_parameter = {
        "max_length": 128,
        "lr": 2e-5,
        "num_epochs": 6,
        "batch_size": 32,
        "dropout_rate": 0.2,
        "weight_decay": 0.001
    }

    # Model trained with BERT algorithm
    bert_model_path = os.path.abspath(os.path.join(
        depends_on,
        "bert_model"
        ))
    bert_log_path = os.path.abspath(os.path.join(
        depends_on,
        "bert_training.log"
        ))

    train_and_evaluate(
        algorithm="bert",
        train_type=None,
        log_path=bert_log_path,
        save_path=bert_model_path,
        max_features=None,
        training_parameters=bert_parameter
    )

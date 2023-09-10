import os
from main_training_function import train_and_evaluate_models

if __name__ == "__main__":
    """
    Examples for training a model with BERT or Random Forest algorithm
    """

    # Path to the model folder for saving
    saving_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "models"
        )
    )

    # Parameter for the training of a model with random forest algorithm
    random_forest_parameter_normal = {
        "n_estimators": 450,
        "max_depth": 80,
        "min_samples_leaf": 2,
        "min_samples_split": 30
    }

    # Grid for training with random forest algorithm with random or grid search
    random_forest_parameter_grid = {
        "n_estimators": [300, 400, 500],
        "max_depth": [70, 80, 90],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [10, 20, 30]
    }

    # Parameter for the training of a model with BERT algorithm
    bert_parameter = {
        "max_length": 128,
        "lr": 2e-5,
        "num_epochs": 6,
        "batch_size": 32,
        "dropout_rate": 0.2,
        "weight_decay": 0.001
    }

    # Model trained with random forest algorithm
    rf_save_path = os.path.join(
        saving_path,
        "random_forest",
        "random_forest_model.joblib"
    )
    rf_log_path = os.path.join(
        saving_path,
        "random_forest",
        "random_forest_training.log"
    )
    train_and_evaluate_models(
        algorithm="random_forest",
        train_type="basic",
        log_path=rf_log_path,
        save_path=rf_save_path,
        max_features=3000,
        training_parameters=random_forest_parameter_normal
    )

    # Model trained with random forest algorithm and random search
    rf_random_save_path = os.path.join(
        saving_path,
        "random_forest",
        "random_forest_model_random.joblib"
    )
    rf_random_log_path = os.path.join(
        saving_path,
        "random_forest",
        "random_forest_training_random.log"
    )
    train_and_evaluate_models(
        algorithm="random_forest",
        train_type="random",
        log_path=rf_random_log_path,
        save_path=rf_random_save_path,
        max_features=5000,
        training_parameters=random_forest_parameter_grid
    )

    # Model trained with random forest algorithm and grid search
    rf_grid_save_path = os.path.join(
        saving_path,
        "random_forest",
        "random_forest_model_grid.joblib"
    )
    rf_grid_log_path = os.path.join(
        saving_path,
        "random_forest",
        "random_forest_training_grid.log"
    )
    train_and_evaluate_models(
        algorithm="random_forest",
        train_type="grid",
        log_path=rf_grid_log_path,
        save_path=rf_grid_save_path,
        max_features=5000,
        training_parameters=random_forest_parameter_grid
    )

    # Model trained with BERT algorithm
    bert_save_path = os.path.join(saving_path, "bert", "bert_model")
    bert_log_path = os.path.join(saving_path, "bert", "bert_training.log")
    train_and_evaluate_models(
        algorithm="bert",
        train_type=None,
        log_path=bert_log_path,
        save_path=bert_save_path,
        max_features=None,
        training_parameters=bert_parameter
    )

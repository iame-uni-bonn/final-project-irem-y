from main_functions import train_and_evaluate

if __name__ == "__main__":
    save_path = '/content/drive/My Drive/OSE_TEST/random_forest_model1.pkl'
    log_path = '/content/drive/My Drive/OSE_TEST/model_training.log'

    random_forest_parameter_normal = {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_leaf": 1,
        "min_samples_split": 2
    }

    random_forest_parameter_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10]
    }

    bert_parameter = {
        "max_length": 128,
        "lr": 2e-5,
        "num_epochs": 4,
        "batch_size": 16,
        "dropout_rate": 0.1,
        "weight_decay": 0.01
    }
    # Model trained with BERT algorithm
    train_and_evaluate(
        algorithm="random_forest",
        train_type="basic",
        log_path=log_path,
        save_path=save_path,
        max_features=5000,
        training_parameters=random_forest_parameter_normal
    )

    # Model trained with BERT algorithm and random search
    train_and_evaluate(
        algorithm="random_forest",
        train_type="random",
        log_path=log_path,
        save_path=save_path,
        max_features=5000,
        training_parameters=random_forest_parameter_normal
    )

    # Model trained with BERT algorithm and grid search
    train_and_evaluate(
        algorithm="random_forest",
        train_type="grid",
        log_path=log_path,
        save_path=save_path,
        max_features=5000,
        training_parameters=random_forest_parameter_normal
    )

    # Model trained with BERT algorithm
    train_and_evaluate(
        algorithm="bert",
        train_type=None,
        log_path=log_path,
        save_path=save_path,
        max_features=None,
        training_parameters=bert_parameter
    )

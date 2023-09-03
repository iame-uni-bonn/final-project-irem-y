from main_functions import train_and_evaluate

if __name__ == "__main__":
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
    # Model trained with random forest algorithm
    rf_save_path = '/content/drive/My Drive/OSE_TEST/random_forest_model.pkl'
    rf_log_path = '/content/drive/My Drive/OSE_TEST/random_forest_training.log'
    train_and_evaluate(
        algorithm="random_forest",
        train_type="basic",
        log_path=rf_log_path,
        save_path=rf_save_path,
        max_features=5000,
        training_parameters=random_forest_parameter_normal
    )

    # Model trained with random forest algorithm and random search
    rf_random_save_path =\
        '/content/drive/My Drive/OSE_TEST/random_forest_model_random.pkl'
    rf_random_log_path =\
        '/content/drive/My Drive/OSE_TEST/random_forest_training_random.log'
    train_and_evaluate(
        algorithm="random_forest",
        train_type="random",
        log_path=rf_random_log_path,
        save_path=rf_random_save_path,
        max_features=5000,
        training_parameters=random_forest_parameter_grid
    )

    # Model trained with random forest algorithm and grid search
    rf_grid_save_path =\
        '/content/drive/My Drive/OSE_TEST/random_forest_model_grid.pkl'
    rf_grid_log_path =\
        '/content/drive/My Drive/OSE_TEST/random_forest_training_grid.log'
    train_and_evaluate(
        algorithm="random_forest",
        train_type="grid",
        log_path=rf_grid_log_path,
        save_path=rf_grid_save_path,
        max_features=5000,
        training_parameters=random_forest_parameter_grid
    )

    # Model trained with BERT algorithm
    bert_save_path = '/content/drive/My Drive/OSE_TEST/bert_model.pkl'
    bert_log_path = '/content/drive/My Drive/OSE_TEST/bert_training.log'
    train_and_evaluate(
        algorithm="bert",
        train_type=None,
        log_path=bert_log_path,
        save_path=bert_save_path,
        max_features=None,
        training_parameters=bert_parameter
    )

from train_models import main_training_pipeline

if __name__ == "__main__":

    training_params_logistic_regression_normal = {
        'C': 0.5,
        'solver': 'lbfgs'
    }

    training_params_logistic_regression_grid = {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    }

    training_params_random_forest_normal = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5
    }

    training_params_random_forest_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    training_params_bert = {
            'max_seq_length': 128,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'epochs': 3
    }

    main_training_pipeline(
        save_path="logistic_regression_normal_model.pth",
        algorithm="logistic_regression",
        train_type="normal",
        train_hyperparameters=training_params_logistic_regression_normal,
        log_file_name="logistic_regression_normal_model_log.txt"
    )

    main_training_pipeline(
        save_path="logistic_regression_random_model.pth",
        algorithm="logistic_regression",
        train_type="random",
        train_hyperparameters=training_params_logistic_regression_grid,
        log_file_name="logistic_regression_random_model_log.txt"
    )
    main_training_pipeline(
        save_path="logistic_regression_grid_model.pth",
        algorithm="logistic_regression",
        train_type="grid",
        train_hyperparameters=training_params_logistic_regression_grid,
        log_file_name="logistic_regression_grid_model_log.txt"
    )

    main_training_pipeline(
        save_path="random_forest_normal_model.pth",
        algorithm="random_forest",
        train_type="normal",
        train_hyperparameters=training_params_random_forest_normal,
        log_file_name="random_forest_normal_model_log.txt"
    )
    main_training_pipeline(
        save_path="random_forest_random_model.pth",
        algorithm="random_forest",
        train_type="random",
        train_hyperparameters=training_params_random_forest_grid,
        log_file_name="random_forest_random_model_log.txt"
    )
    main_training_pipeline(
        save_path="random_forest_grid_model.pth",
        algorithm="random_forest",
        train_type="grid",
        train_hyperparameters=training_params_random_forest_grid,
        log_file_name="random_forest_grid_model_log.txt"
    )

    main_training_pipeline(
        save_path="bert_model.pth",
        algorithm="bert",
        train_type=None,
        train_hyperparameters=training_params_bert,
        log_file_name="logfile.txt"
    )

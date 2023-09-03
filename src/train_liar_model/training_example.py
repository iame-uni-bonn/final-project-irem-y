from main_functions import train_and_evaluate

if __name__ == "__main__":
    save_path = '/content/drive/My Drive/OSE_TEST/random_forest_model1.pkl'
    log_path = '/content/drive/My Drive/OSE_TEST/model_training.log'

    random_forest_parameter = {
        "max_features": 5000,
        "n_estimators": 100
    }

    bert_parameter = {
        "max_length": 128,
        "lr": 1e-5,
        "num_epochs": 3,
        "batch_size": 8
    }

    train_and_evaluate(
        "random_forest",
        log_path,
        save_path,
        random_forest_parameter
    )

    train_and_evaluate(
        "bert",
        log_path,
        save_path,
        bert_parameter
    )

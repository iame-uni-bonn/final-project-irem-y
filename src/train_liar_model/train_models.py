import numpy as np
import sys
import torch
from datasets import load_dataset
from bert_functions import train_and_evaluate_bert_model
from LR_and_RF_functions import train_and_evaluate_classification_model


def load_liar_data():
    """
    Load the LIAR dataset using the Hugging Face datasets library.

    Returns:
        train_data (Dataset): Training data
        test_data (Dataset): Test data
        validation_data (Dataset): Validation data
    """
    dataset = load_dataset("liar")
    return dataset["train"], dataset["test"], dataset["validation"]


def preprocess_data(dataset):
    """
    Preprocess the dataset to extract statements and corresponding labels.

    Args:
        dataset (Dataset): A dataset containing statements and labels.

    Returns:
        statements (list): List of statement texts.
        statement_labels (list): List of corresponding label indices.
    """
    statements = dataset["statement"]
    statement_labels = dataset["label"]
    return statements, statement_labels


def main_training_pipeline(
        save_path,
        algorithm,
        train_type,
        train_hyperparameters,
        log_file_name
        ):
    """
    Main training pipeline that orchestrates model training and evaluation.

    Args:
        save_path (str): Path to save the trained model.
        algorithm (str): Algorithm name ("logistic_regression",
        "random_forest", or "bert").
        train_type (str): Type of training ("normal", "grid", or "random").
        train_hyperparameters (dict): Hyperparameters for the training.
        log_file_name (str): Name of the log file.

    """
    # Check if input parameter are correct
    logistic_reg_parameter = ["C", "solver"]
    random_forest_parameter = [
        'n_estimators',
        'max_depth',
        'min_samples_split'
    ]
    bert_parameter = [
        'max_seq_length',
        'batch_size',
        'learning_rate',
        'epochs'
    ]

    if type(save_path) != str:
        print(f"save_path must be a string. You gave {type(save_path)}.")
        return

    if type(algorithm) != str:
        print(f"algorithm has to be a string. You gave {type(algorithm)}.")
        return

    if algorithm not in ("logistic_regression", "random_forest", "bert"):
        print(
            f"algorithm has to be one of the following values:"
            f" logistic_regression, random_forest or bert. "
            f"You gave {algorithm}."
        )
        return

    if type(train_type) != str:
        print(f"train_type has to be a string. You gave {type(train_type)}.")
        return

    if algorithm != "bert" and train_type not in ("normal", "grid", "random"):
        print(
            f"train_type hast to be one of the following values:"
            f" normal, grid or random. "
            f" You gave {train_type}"
        )
        return

    if algorithm == "logistic_regression":
        if len(list(train_hyperparameters.keys())) != 2:
            print(
                f"train_hyperparameters has to have four parameter. You gave "
                f"{len(train_hyperparameters)}."
            )
            return
        if set(train_hyperparameters.keys()) != set(logistic_reg_parameter):
            print(
                f"train_hyperparameters has to have the following keys: "
                f"{logistic_reg_parameter}. "
                f"You gave {list(train_hyperparameters.keys())}."
            )
            return

    if algorithm == "random_forest":
        if len(list(train_hyperparameters.keys())) != 3:
            print(
                f"train_hyperparameters has to have four parameter. You gave "
                f"{len(train_hyperparameters)}."
            )
            return
        if set(train_hyperparameters.keys()) != set(random_forest_parameter):
            print(
                f"train_hyperparameters has to have the following keys:"
                f"{random_forest_parameter}"
                f"You gave {list(train_hyperparameters.keys())}."
            )
            return

    if algorithm == "bert":
        if len(list(train_hyperparameters.keys())) != 4:
            print(
                f"train_hyperparameters has to have four parameter. You gave "
                f"{len(train_hyperparameters)}."
            )
            return
        if set(train_hyperparameters.keys()) != set(bert_parameter):
            print(
                f"train_hyperparameters has to have the following keys: "
                f"{bert_parameter}. "
                f"You gave {list(train_hyperparameters.keys())}."
            )
            return

    # Set a fixed seed for reproducibility
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    # Redirect stdout to the log file
    log_file = open(log_file_name, "w")
    sys.stdout = log_file

    # Determine the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load LIAR dataset
    train_data, test_data, validation_data = load_liar_data()
    train_statements, train_statement_labels = preprocess_data(train_data)
    test_statements, test_statement_labels = preprocess_data(test_data)
    validation_statements, validation_statement_labels = preprocess_data(
        validation_data
    )

    # Train and evaluate the specified model
    if algorithm == "bert":
        print("Training and evaluating BERT model...")
        train_and_evaluate_bert_model(
            train_statements,
            train_statement_labels,
            validation_statements,
            validation_statement_labels,
            test_statements,
            test_statement_labels,
            train_hyperparameters,
            device,
            save_path
        )
    elif algorithm in ("logistic_regression", "random_forest"):

        train_and_evaluate_classification_model(
            algorithm=algorithm,
            train_style=train_type,
            save_path=save_path,
            training_params=train_hyperparameters,
            y_train=train_statement_labels,
            train_statements=train_statements,
            y_validation=validation_statement_labels,
            validation_statements=validation_statements,
            y_test=test_statement_labels,
            test_statements=test_statements
        )

    log_file.close()
    sys.stdout = sys.__stdout__

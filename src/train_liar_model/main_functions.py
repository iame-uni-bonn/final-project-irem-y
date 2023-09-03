import pandas as pd
import torch
import numpy as np
from datasets import load_dataset
import sys
import random
from bert_functions import train_and_evaluate_bert_classifier
from random_forest_functions import train_and_evaluate_random_forest_classifier

# Set seeds for reproducibility
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# If using CUDA (GPU), set a seed for CUDA devices as well
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


def setup_logging(log_file_name):
    """
    Set up logging and redirect standard output to a log file.

    Args:
        log_file_name (str): Name of the log file.

    Returns:
        file: The opened log file.
    """
    # Open the log file for writing
    log_file = open(log_file_name, "w")
    # Redirect standard output to the log file
    sys.stdout = log_file
    return log_file


def close_logging(log_file):
    """
    Close the log file and reset standard output.

    Args:
        log_file (file): The log file to be closed.
    """
    log_file.close()
    sys.stdout = sys.__stdout__


def load_liar_dataset():
    """
    Load the LIAR dataset using the Hugging Face datasets library.

    Returns:
        tuple: A tuple containing train, test, and validation dataset splits.
    """
    try:
        dataset = load_dataset("liar")
        return dataset["train"], dataset["test"], dataset["validation"]
    except Exception as e:
        print("Error loading the dataset:", str(e))
        raise


def prepare_dataframe(train_data, test_data, validation_data):
    """
    Convert dataset splits into Pandas DataFrames.

    Args:
        train_data (dict): Training dataset split.
        test_data (dict): Test dataset split.
        validation_data (dict): Validation dataset split.

    Returns:
        tuple: A tuple containing train, test, and validation DataFrames.
    """
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    validation_df = pd.DataFrame(validation_data)
    return train_df, test_df, validation_df


def check_input_parameter(
        algorithm,
        train_type,
        log_path,
        save_path,
        max_features,
        training_parameters
        ):
    """
    Check if the provided parameter are correct.

    Args:
        algorithm (str): The algorithm to use, 'bert' or 'random_forest'.
        train_type (str): Style of training ("normal", "grid", or "random").
        log_path (str): Path to the log file.
        save_path (str): Path to save the trained model.
        max_features (int): Maximum number of features for TF-IDF for
        random forest training.
        training_parameters (dict): Dictionary containing algorithm-specific
        parameters.

    Returns:
        boolean: A boolean if the parameter are correct or not
    """
    possible_algorithm = ["random_forest", "bert"]
    possible_train_types = ["basic", "grid", "random"]
    possible_bert_parameters = [
        "max_length",
        "lr",
        "num_epochs",
        "batch_size",
        "dropout_rate",
        "weight_decay"
    ]
    possible_random_forest_parameters = [
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "min_samples_split"
    ]
    if algorithm not in possible_algorithm:
        print(
            f"algorithm hast to be one of the following values: "
            f"{possible_algorithm}. You have entered: {algorithm}"
        )
        return False
    if train_type not in possible_train_types and algorithm == "random_forest":
        print(
            f"train_type has to be one of the following values: "
            f"{possible_train_types}. You have entered {train_type}."
        )
        return False
    if type(log_path) != str:
        print(
            f"log_path hast to be a string. "
            f"You have entered a {type(log_path)}."
        )
        return False
    if type(save_path) != str:
        print(
            f"log_path hast to be a string. "
            f"You have entered a {type(save_path)}."
        )
        return False
    if type(training_parameters) != dict:
        print(
            f"training_parameters hast to be a dictionary. "
            f"You have entered a {type(training_parameters)}."
        )
        return False
    if algorithm == "bert":
        if set(training_parameters.keys()) != set(possible_bert_parameters):
            print(
                f"training_parameters has to have the following keys: "
                f"{possible_bert_parameters}. "
                f"You have entered {training_parameters.keys()}."
            )
            return False
    elif algorithm == "random_forest":
        if type(max_features) != int:
            print(
                f"max_features hast to be a integer. "
                f"You have entered a {type(max_features)}."
            )
        if set(training_parameters.keys()) !=\
                set(possible_random_forest_parameters):
            print(
                f"training_parameters has to have the following keys: "
                f"{possible_bert_parameters}. "
                f"You have entered {training_parameters.keys()}."
            )
            return False
    return True


def train_and_evaluate(
        algorithm,
        train_type,
        log_path,
        save_path,
        max_features,
        training_parameters
        ):
    """
    Train and evaluate a classifier using the specified algorithm.

    Args:
        algorithm (str): The algorithm to use, 'bert' or 'random_forest'.
        train_type (str): Style of training ("normal", "grid", or "random")
        for random forest.
        log_path (str): Path to the log file.
        save_path (str): Path to save the trained model.
        max_features (int): Maximum number of features for TF-IDF for
        random forest training.
        training_parameters (dict): Dictionary containing algorithm-specific
        parameters.
    """
    # Set up logging
    log_file = setup_logging(log_path)

    try:
        # Check input parameter
        correct_parameter = check_input_parameter(
            algorithm,
            train_type,
            log_path,
            save_path,
            max_features,
            training_parameters
        )
        if correct_parameter is False:
            return

        print(f"Training model with: {algorithm}")
        print(f"The training parameter are: {training_parameters}")
        if algorithm == "bert":
            print("")
            train_and_evaluate_bert_classifier(
                save_path=save_path,
                max_length=training_parameters["max_length"],
                lr=training_parameters["lr"],
                num_epochs=training_parameters["num_epochs"],
                batch_size=training_parameters["batch_size"],
                dropout_rate=training_parameters["dropout_rate"],
                weight_decay=training_parameters["weight_decay"]
            )
        elif algorithm == "random_forest":
            print(f"Maximum number of features for TF-IDF are: {max_features}")
            print(f"The train style is: {train_type}")
            print("")
            train_and_evaluate_random_forest_classifier(
                save_path=save_path,
                train_type=train_type,
                max_features=max_features,
                random_state=RANDOM_STATE,
                trainings_parameter=training_parameters
            )

    except Exception as e:
        print("An error occurred:", str(e))

    # Close the log file
    close_logging(log_file)

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


def train_and_evaluate(algorithm, log_path, save_path, training_parameters):
    """
    Train and evaluate a classifier using the specified algorithm.

    Args:
        algorithm (str): The algorithm to use, 'bert' or 'random_forest'.
        log_path (str): Path to the log file.
        save_path (str): Path to save the trained model.
        training_parameters (dict): Dictionary containing algorithm-specific
        parameters.
    """
    # Set up logging
    log_file = setup_logging(log_path)

    try:
        if algorithm == "bert":
            train_and_evaluate_bert_classifier(
                save_path,
                training_parameters["max_length"],
                training_parameters["lr"],
                training_parameters["num_epochs"],
                training_parameters["batch_size"]
            )
        elif algorithm == "random_forest":
            train_and_evaluate_random_forest_classifier(
                save_path,
                training_parameters["max_features"],
                training_parameters["n_estimators"],
                RANDOM_STATE
            )
        else:
            print(f"Only 'bert' and 'random_forest' are supported algorithms. "
                  f"You have entered: {algorithm}")

    except Exception as e:
        print("An error occurred:", str(e))

    # Close the log file
    close_logging(log_file)

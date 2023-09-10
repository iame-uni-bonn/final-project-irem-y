"""
Help functions that are used in multiple other functions.

Functions:
    - latex_to_pdf(out_path, file_path)
    - setup_logging(log_file_name)
    - close_logging(log_file)
    - load_liar_dataset()
    - prepare_dataframe(train_data, test_data, validation_data)
    - check_input_parameter(
        algorithm,
        train_type,
        log_path,
        save_path,
        max_features,
        training_parameters
        )
"""

import pandas as pd
import os
import subprocess
import sys
from datasets import load_dataset


def latex_to_pdf(out_path, file_path):
    """
    Run Latex file to create PDF.

    Args:
        out_path (str): Path to save the output
        file_path (str): Path of the Excel file

    Returns:
        None
    """
    try:
        run_latex = "pdflatex -output-directory=" + out_path + " " + file_path
        subprocess.run(run_latex, stdout=subprocess.DEVNULL)
        print(
            f"PDF for {os.path.basename(file_path)} successfully created in:"
            f" {out_path}"
        )
    except Exception as e:
        print("Error compiling the lastex file:", str(e))
        raise


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

    Returns:
        None
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

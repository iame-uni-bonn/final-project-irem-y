import torch
import numpy as np
import random
from bert_functions import train_and_evaluate_bert_classifier
from general_functions import (
    setup_logging,
    check_input_parameter,
    close_logging
)
from random_forest_functions import train_and_evaluate_random_forest_classifier


def train_and_evaluate_models(
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
            accuracy, report = train_and_evaluate_bert_classifier(
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
            accuracy, report = train_and_evaluate_random_forest_classifier(
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

    return accuracy, report

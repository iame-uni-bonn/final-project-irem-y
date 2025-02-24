"""
A function to run the whole program and the function call for this function.

Also create the PDFs of the Latex files.

Functions:
    - train_and_compare_models(rf_name, bert_name, hf_name)
"""

import os
from general_functions import latex_to_pdf
from main_training_function import (
    train_and_evaluate_models
)
from compare_functions import (
    load_and_compare_models
)


def train_and_compare_models(rf_name, bert_name, hf_name):
    """
    Train two models and compare them with a model from Huggingface.

    Args:
        rf_name (str): Name of the Random Forest model
        text_features (str): Name of the Bert model
        other_features (str): Name of the Huggingface model

    Returns:
        None
    """
    # Path to the model folder for saving
    saving_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "models"
        )
    )

    # Create saving folder if not existing yet
    rf_path = os.path.join(saving_path, "random_forest")
    if not os.path.exists(rf_path):
        os.makedirs(rf_path)
    bert_path = os.path.join(saving_path, "bert")
    if not os.path.exists(bert_path):
        os.makedirs(bert_path)

    # Parameter for the training of a model with random forest algorithm
    random_forest_parameter_normal = {
        "n_estimators": 450,
        "max_depth": 80,
        "min_samples_leaf": 2,
        "min_samples_split": 30
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
        rf_path,
        rf_name+".joblib"
    )
    rf_log_path = os.path.join(
        rf_path,
        rf_name+".log"
    )

    # Check if the paths are free
    if os.path.isfile(rf_save_path) is True:
        print(
            f"The model {rf_save_path} already exist. "
            f"Please use a different name as {rf_name}."
        )
    if os.path.isfile(rf_log_path) is True:
        print(
            f"The log file {rf_log_path} already exist. "
            f"Please use a different name as {rf_name}."
        )

    rf_accuracy, rf_report = train_and_evaluate_models(
        algorithm="random_forest",
        train_type="basic",
        log_path=rf_log_path,
        save_path=rf_save_path,
        max_features=3000,
        training_parameters=random_forest_parameter_normal
    )
    print(f"Random forest {rf_name} successfully trained.")
    # Model trained with BERT algorithm
    bert_save_path = os.path.join(bert_path, bert_name)
    bert_log_path = os.path.join(bert_path, "bert", bert_name+".log")

    # Check if the paths are free
    if os.path.isfile(bert_save_path) is True:
        print(
            f"The model {bert_save_path} already exist. "
            f"Please use a different name as {bert_name}."
        )
    if os.path.isfile(bert_log_path) is True:
        print(
            f"The log file {bert_log_path} already exist. "
            f"Please use a different name as {bert_name}."
        )

    bert_accuracy, bert_report = train_and_evaluate_models(
        algorithm="bert",
        train_type=None,
        log_path=bert_log_path,
        save_path=bert_save_path,
        max_features=None,
        training_parameters=bert_parameter
    )
    print(f"BERT {bert_name} successfully trained.")

    load_and_compare_models(
        [
            (rf_name, rf_accuracy, rf_report),
            (bert_name, bert_accuracy, bert_report)
        ],
        hf_name
    )


if __name__ == "__main__":

    # Run the program
    train_and_compare_models(
        "random_forest_model",
        "bert_model",
        "Jawaher/LIAR-fake-news-roberta-base"
    )

    # Get the path of the docs folder
    docs_path = os.path.join(
        os.path.abspath(__file__),
        "..",
        "..",
        "..",
        "docs"
    )

    # Create the PDFs from the Latex file
    question_path = os.path.join(
        docs_path,
        "25_Questions.tex"
    )

    latex_to_pdf(
        os.path.abspath(docs_path),
        os.path.abspath(question_path)
    )

    project_documentation_path = os.path.join(
        docs_path,
        "project_documentation.tex"
    )

    latex_to_pdf(
        os.path.abspath(docs_path),
        os.path.abspath(project_documentation_path)
    )

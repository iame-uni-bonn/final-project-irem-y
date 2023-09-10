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
    Train two models and compare them with a model from Huggingface

    Args:
        rf_name (str): Name of the Random Forest model
        text_features (str): Name of the Bert model
        other_features (str): Name of the Huggingface model
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

    # Parameter for the training of a model with random forest algorithm
    random_forest_parameter_normal = {
        "n_estimators": 450,
        "max_depth": 80,
        "min_samples_leaf": 2,
        "min_samples_split": 30
    }

    # Parameter for the training of a model with BERT algorithm
    bert_parameter = {
        "max_length": 64,
        "lr": 2e-5,
        "num_epochs": 1,
        "batch_size": 16,
        "dropout_rate": 0.2,
        "weight_decay": 0
    }

    # Model trained with random forest algorithm
    rf_save_path = os.path.join(
        saving_path,
        "random_forest",
        rf_name+".joblib"
    )
    rf_log_path = os.path.join(
        saving_path,
        "random_forest",
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
    print(f"Rnadom forest {rf_name} successfully trained.")
    # Model trained with BERT algorithm
    bert_save_path = os.path.join(saving_path, "bert", bert_name)
    bert_log_path = os.path.join(saving_path, "bert", bert_name+".log")

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
    """
    train_and_compare_models(
        "random_forest_model",
        "bert_model",
        "Jawaher/LIAR-fake-news-roberta-base"
    )
    """
    docs_path = os.path.join(
        os.path.abspath(__file__),
        "..",
        "..",
        "..",
        "docs"
    )
    question_path = os.path.join(
        docs_path,
        "25_Questions.tex"
    )

    # Create PDFs
    latex_to_pdf(docs_path, question_path)

    project_documentation = os.path.join(
        docs_path,
        "project_documentation.tex"
    )

    latex_to_pdf(docs_path, project_documentation)

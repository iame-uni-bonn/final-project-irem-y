import pickle
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          BertModel,
                          BertTokenizer)


def load_random_forest_model(path):
    """
    Load a Random Forest model from a pickle file.

    Args:
        path (str): The file path to the saved Random Forest model in pickle
        format.

    Returns:
        object: The loaded Random Forest model.
    """
    with open(path, 'rb') as file:
        loaded_model = pickle.load(file)

    return loaded_model


def load_model_hugginface(model_name):
    """
    Load a pre-trained Hugging Face model and tokenizer.

    Args:
        model_name (str): The name of the pre-trained model to load.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_local_bert(path):
    """
    Load a locally saved BERT model and its tokenizer.

    Args:
        path (str): The directory path containing the saved BERT model files.

    Returns:
        tuple: A tuple containing the loaded BERT model and tokenizer.
    """
    model = BertModel.from_pretrained(path)
    tokenizer = BertTokenizer.from_pretrained(path)
    return model, tokenizer

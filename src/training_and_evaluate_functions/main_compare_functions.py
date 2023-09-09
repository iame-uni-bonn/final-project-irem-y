import torch
from utils.general_functions import (
    load_liar_dataset,
    prepare_dataframe
)
from bert_functions import (
    evaluate_bert_model_performance,
    create_data_loader,
    prepare_features,
    tokenize_text_data
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)


def load_model_huggingface(model_name):
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


def evaluate_bert_model(bert_model, tokenizer, max_length, batch_size):
    """
    Evaluate a BERT-based classifier.

    Args:
        bert_model (object): The BERT model to be evaluated.
        tokenizer (object): The tokenizer of the BERT model.
        max_length (int): Maximum sequence length for tokenization.
        batch_size (int): Batch size for test data.

    Returns:
        tuple: A tuple containing test accuracy and classification report.

    """
    # Prepare the feature matrix X_train
    text_feature_columns = [
        'statement',
        'subject',
        'speaker',
        'job_title',
        'state_info',
        'party_affiliation',
        'context'
    ]

    # Other non-text features
    other_feature_columns = [
        'barely_true_counts',
        'false_counts',
        'half_true_counts',
        'mostly_true_counts',
        'pants_on_fire_counts'
    ]
    # Load the LIAR dataset
    train_data, test_data, validation_data = load_liar_dataset()

    # Prepare data
    train_df, test_df, validation_df = prepare_dataframe(
        train_data,
        test_data,
        validation_data
    )
    X_test_text, X_test_other = prepare_features(
        test_df,
        text_feature_columns,
        other_feature_columns
    )

    # Tokenize data
    test_encodings = tokenize_text_data(
        X_test_text,
        tokenizer,
        max_length
    )

    test_labels = torch.tensor(list(test_df['label']))

    # Create data loaders for testing
    test_loader = create_data_loader(
        test_encodings,
        X_test_other,
        test_labels,
        batch_size
    )

    # Set up model and train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Evaluate the model on the test set
    test_accuracy, test_classification_rep = evaluate_bert_model_performance(
        bert_model,
        test_loader,
        device
    )

    return test_accuracy, test_classification_rep


def extract_metrics_from_report(report):
    """
    Extract precision, recall, and F1-score from a classification report.

    Args:
        report (str): The classification report string.

    Returns:
        tuple: A tuple containing precision, recall, and F1-score as floats.

    """
    substring_weighted_avg = report[report.find("weighted avg"):]
    parameter_list = []
    for x in range(3):
        number_pos = substring_weighted_avg.find(".")
        parameter_list.append(
            float(substring_weighted_avg[number_pos-1:number_pos+4])
        )
        substring_weighted_avg = substring_weighted_avg[number_pos+5:]
    return tuple(parameter_list)


def rank_models(model_data):
    """
    Rank models based on accuracy, precision, recall, and F1-score.

    Args:
        model_data (list of tuple): A list of tuples containing model name,
        accuracy, and classification report.

    """
    # Initialize dictionaries to store metrics
    accuracy_scores = {}
    precision_scores = {}
    recall_scores = {}
    f1_scores = {}

    # Calculate metrics for each model
    for model_name, accuracy, report in model_data:
        precision, recall, f1 = extract_metrics_from_report(report)

        accuracy_scores[model_name] = accuracy
        precision_scores[model_name] = precision
        recall_scores[model_name] = recall
        f1_scores[model_name] = f1

    # Rank the models based on accuracy
    ranked_models_accuracy = sorted(
        model_data, key=lambda x: x[1], reverse=True
    )

    # Rank the models based on precision
    ranked_models_precision = sorted(
        model_data, key=lambda x: precision_scores[x[0]], reverse=True
    )

    # Rank the models based on recall
    ranked_models_recall = sorted(
        model_data, key=lambda x: recall_scores[x[0]], reverse=True
    )

    # Rank the models based on F1-score
    ranked_models_f1 = sorted(
        model_data, key=lambda x: f1_scores[x[0]], reverse=True
    )

    # Print the ranked models for each metric
    print("Ranking based on Accuracy:")
    for rank, (model_name, accuracy, _) in enumerate(ranked_models_accuracy,
                                                     start=1):
        print(f"Rank {rank}: {model_name}, Accuracy: {accuracy:.2f}")

    print("\nRanking based on Precision:")
    for rank, (model_name, _, _) in enumerate(ranked_models_precision,
                                              start=1):
        print(
            f"Rank {rank}: {model_name}, Precision: "
            f"{precision_scores[model_name]:.2f}"
        )

    print("\nRanking based on Recall:")
    for rank, (model_name, _, _) in enumerate(ranked_models_recall, start=1):
        print(
            f"Rank {rank}: {model_name}, Recall: "
            f"{recall_scores[model_name]:.2f}"
        )

    print("\nRanking based on F1-score:")
    for rank, (model_name, _, _) in enumerate(ranked_models_f1, start=1):
        print(
            f"Rank {rank}: {model_name}, F1-score: "
            f"{f1_scores[model_name]:.2f}"
        )


def load_and_compare_models(result_list, hugginface_model):
    """
    Load a Hugging Face model, evaluate its, and compare it to other models.

    Args:
        result_list (list of tuple): A list of tuples containing model name,
        accuracy, and classification report.
        hugginface_model (str): The name of the Hugging Face model to load and
        evaluate.

    """
    huggingface_model, huggingface_tokenizer = load_model_huggingface(
        hugginface_model
    )
    huggingface_bert_accuracy, huggingface_bert_report = evaluate_bert_model(
        huggingface_model,
        huggingface_tokenizer,
        128,
        32
    )
    huggingface_result = (
        hugginface_model,
        huggingface_bert_accuracy,
        huggingface_bert_report
    )

    result_list.append(huggingface_result)
    rank_models(result_list)

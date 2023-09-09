import torch
from train_liar_model.general_functions import (
    load_liar_dataset,
    prepare_dataframe
)
from train_liar_model.random_forest_functions import (
    evaluate_random_forest_classifier_performance
)
from train_liar_model.bert_functions import (
    evaluate_bert_model_performance,
    create_data_loader,
    prepare_features,
    tokenize_text_data
)


def evaluate_rf_model(random_forest_model, max_features):
    """
    Evaluate a Random Forest classifier.

    Args:
        random_forest_model (object): The Random Forest model to be evaluated.
        max_features (int): Maximum number of features for TF-IDF.

    Returns:
        tuple: A tuple containing accuracy and classification report.

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

    # Evaluate the trained model
    accuracy, classification_rep =\
        evaluate_random_forest_classifier_performance(
            random_forest_model,
            test_df,
            text_feature_columns,
            other_feature_columns,
            max_features
        )

    return accuracy, classification_rep


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

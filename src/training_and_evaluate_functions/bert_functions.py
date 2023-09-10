"""
Function to train a model to recognize fake news with the BERT algorithm.

Functions:
    - prepare_features(df, text_features, other_features)
    - tokenize_text_data(text_feature_matrix, tokenizer, max_length)
    - create_data_loader(
        train_encodings,
        X_train_other,
        train_labels,
        batch_size
        )
    - save_model_and_tokenizer(model, tokenizer, save_path)
    - initialize_bert_model(model_name, num_labels, device, dropout_rate)
    - train_bert_model(
        model_name,
        num_labels,
        lr,
        num_epochs,
        dropout_rate,
        weight_decay,
        train_loader,
        val_loader,
        device
        )
    - evaluate_bert_model_performance(model, dataloader, device)
    - train_and_evaluate_bert_classifier(
        save_path,
        max_length,
        lr,
        num_epochs,
        batch_size,
        dropout_rate,
        weight_decay
        )
"""

import torch
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, logging
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from general_functions import load_liar_dataset, prepare_dataframe


def prepare_features(df, text_features, other_features):
    """
    Prepare the text and other feature matrices for training.

    Args:
        df (DataFrame): The input DataFrame containing both text and other
        features.
        text_features (list): List of columns containing text features.
        other_features (list): List of columns containing non-text features.

    Returns:
        tuple: A tuple containing two elements:
            - text_feature_matrix (list): List of text feature matrices.
            - other_feature_matrix (numpy.ndarray): Matrix of other feature
            values.
    """
    text_feature_matrix = df[text_features].astype(str).values.sum(axis=1)
    text_feature_matrix = [" ".join(row) for row in text_feature_matrix]
    other_feature_matrix = df[other_features].values
    return text_feature_matrix, other_feature_matrix


def tokenize_text_data(text_feature_matrix, tokenizer, max_length):
    """
    Tokenize the text data using the provided BERT tokenizer.

    Args:
        text_feature_matrix (list): List of text feature data to be tokenized.
        tokenizer (BertTokenizer): The BERT tokenizer.
        max_length (int): Maximum sequence length.

    Returns:
        dict: Tokenized encodings containing 'input_ids' and 'attention_mask'.
    """
    try:
        encodings = tokenizer(
            text_feature_matrix,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return encodings
    except Exception as e:
        print("Error tokenizing data:", str(e))
        raise


def create_data_loader(
        train_encodings,
        X_train_other,
        train_labels,
        batch_size
        ):
    """
    Create a PyTorch data loader for training data.

    Args:
        train_encodings (dict): Tokenized encodings containing 'input_ids' and
        'attention_mask'.
        X_train_other (numpy.ndarray): Matrix of other feature values.
        train_labels (Tensor): Training labels.
        batch_size (int): Batch size.

    Returns:
        DataLoader: PyTorch data loader for training data.
    """
    try:
        # Ensure the input sizes are compatible
        assert (
            len(train_encodings['input_ids']) ==
            len(X_train_other) ==
            len(train_labels)
        )

        # Create PyTorch data loader for training data
        train_other_features = torch.tensor(X_train_other, dtype=torch.float32)
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            train_other_features,
            train_labels
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        return train_loader

    except Exception as e:
        print("Error creating data loader:", str(e))
        raise


def save_model_and_tokenizer(model, tokenizer, save_path):
    """
    Save the BERT model and tokenizer to the specified path.

    Args:
        model (BertForSequenceClassification): The BERT model.
        tokenizer (BertTokenizer): The BERT tokenizer.
        save_path (str): Path to save the model and tokenizer.

    Returns:
        None
    """
    try:
        # Save the model and tokenizer to the specified path
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")
    except Exception as e:
        print("Error saving model and tokenizer:", str(e))
        raise


def initialize_bert_model(model_name, num_labels, device, dropout_rate):
    """
    Initialize a BERT-based sequence classification model.

    Args:
        model_name (str): BERT model name.
        num_labels (int): Number of labels for classification.
        device (str): Device for model placement.
        dropout_rate (float): Dropout rate for BERT layers to prevent
        overfitting.

    Returns:
        BertForSequenceClassification: Initialized BERT model.
    """

    # Only show errors
    logging.set_verbosity_error()

    # Import pre-trained bert model and set it up
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        hidden_dropout_prob=dropout_rate,
        attention_probs_dropout_prob=dropout_rate
    )
    model.to(device)
    return model


def train_bert_model(
        model_name,
        num_labels,
        lr,
        num_epochs,
        dropout_rate,
        weight_decay,
        train_loader,
        val_loader,
        device
        ):
    """
    Initialize and train the BERT-based sequence classification model.

    Args:
        model_name (str): BERT model name.
        num_labels (int): Number of labels for classification.
        lr (float): Learning rate.
        num_epochs (int): Number of training epochs.
        dropout_rate (float): Dropout rate for BERT layers to prevent
        overfitting.
        weight_decay (float): Weight decay for the optimizer to prevent
        overfitting.
        train_loader (DataLoader): PyTorch data loader for training data.
        val_loader (DataLoader): PyTorch data loader for validation data.
        device (str): Device for model placement.

    Returns:
        BertForSequenceClassification: Trained BERT model.
    """
    try:
        model = initialize_bert_model(
            model_name,
            num_labels,
            device,
            dropout_rate
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            total_batches = len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs} - Training...")

            for batch_idx, batch in enumerate(train_loader, start=1):
                batch = [item.to(device) for item in batch]
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'input_other_features': batch[2],
                    'labels': batch[3]
                }
                optimizer.zero_grad()
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=inputs['labels']
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    avg_batch_loss = total_loss / batch_idx
                    print(
                        f"Batch {batch_idx}/{total_batches}"
                        f" - Loss: {avg_batch_loss:.4f}"
                    )

            avg_epoch_loss = total_loss / total_batches
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Average Training Loss: {avg_epoch_loss:.4f}"
            )

            # Validate the model after each epoch
            val_accuracy, val_classification_rep = \
                evaluate_bert_model_performance(
                    model,
                    val_loader,
                    device
                )
            print(
                f"Validation Accuracy after Epoch {epoch + 1}: "
                f"{val_accuracy:.4f}"
            )
            print(
                "Validation Classification Report:\n",
                val_classification_rep
            )

        return model

    except Exception as e:
        print("Error setting up model and training:", str(e))
        raise


def evaluate_bert_model_performance(model, dataloader, device):
    """
    Evaluate the model's performance on validation or test data.

    Args:
        model (BertForSequenceClassification): Trained BERT model.
        dataloader (DataLoader): PyTorch data loader for evaluation data.
        device (str): Device for model placement.

    Returns:
        tuple: A tuple containing accuracy and classification report.
    """
    try:
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                batch = [item.to(device) for item in batch]
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'input_other_features': batch[2],
                    'labels': batch[3]
                }
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                # Check the attribute name in outputs for logits
                if 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = outputs['last_hidden_state']
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch[3].cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        classification_rep = classification_report(all_labels, all_preds)
        return accuracy, classification_rep

    except Exception as e:
        print("Error during evaluation:", str(e))
        raise


def train_and_evaluate_bert_classifier(
        save_path,
        max_length,
        lr,
        num_epochs,
        batch_size,
        dropout_rate,
        weight_decay
        ):
    """
    Train and evaluate a BERT-based classifier.

    Args:
        save_path (str): Path to save the trained model and tokenizer.
        max_length (int): Maximum sequence length for tokenization.
        lr (float): Learning rate for training.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        dropout_rate (float): Dropout rate for BERT layers to prevent
        overfitting.
        weight_decay (float): Weight decay for the optimizer to prevent
        overfitting.

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
    X_train_text, X_train_other = prepare_features(
        train_df,
        text_feature_columns,
        other_feature_columns
    )
    X_val_text, X_val_other = prepare_features(
        validation_df,
        text_feature_columns,
        other_feature_columns
    )
    X_test_text, X_test_other = prepare_features(
        test_df,
        text_feature_columns,
        other_feature_columns
    )

    # Tokenize data
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_encodings = tokenize_text_data(
        X_train_text,
        tokenizer,
        max_length
    )
    val_encodings = tokenize_text_data(
        X_val_text,
        tokenizer,
        max_length
    )
    test_encodings = tokenize_text_data(
        X_test_text,
        tokenizer,
        max_length
    )

    train_labels = torch.tensor(list(train_df['label']))
    val_labels = torch.tensor(list(validation_df['label']))
    test_labels = torch.tensor(list(test_df['label']))

    # Create data loaders for training, validation, and test
    train_loader = create_data_loader(
        train_encodings,
        X_train_other,
        train_labels,
        batch_size
    )
    val_loader = create_data_loader(
        val_encodings,
        X_val_other,
        val_labels,
        batch_size
    )
    test_loader = create_data_loader(
        test_encodings,
        X_test_other,
        test_labels,
        batch_size
    )

    # Set up model and train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = 6
    trained_model = train_bert_model(
        model_name,
        num_labels,
        lr,
        num_epochs,
        dropout_rate,
        weight_decay,
        train_loader,
        val_loader,
        device
    )

    # Evaluate the model on the validation set
    val_accuracy, val_classification_rep = evaluate_bert_model_performance(
        trained_model,
        val_loader,
        device
    )
    print("Validation Accuracy:", val_accuracy)
    print("Validation Classification Report:\n", val_classification_rep)

    # Evaluate the model on the test set
    test_accuracy, test_classification_rep = evaluate_bert_model_performance(
        trained_model,
        test_loader,
        device
    )
    print("Test Accuracy:", test_accuracy)
    print("Test Classification Report:\n", test_classification_rep)

    # Save the trained BERT model
    save_model_and_tokenizer(trained_model, tokenizer, save_path)
    return test_accuracy, test_classification_rep

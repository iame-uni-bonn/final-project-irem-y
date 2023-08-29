import torch
from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn
import torch.optim as optim
from transformers import (BertTokenizer,
                          BertForSequenceClassification,
                          AdamW,
                          get_linear_schedule_with_warmup)


def prepare_bert_inputs(
        tokenizer,
        statements,
        statement_labels,
        max_seq_length
        ):
    """
    Prepare inputs for BERT model by tokenizing and formatting the data.

    Args:
        tokenizer (BertTokenizer): Pre-trained BERT tokenizer.
        statements (list): List of statement texts.
        statement_labels (list): List of corresponding label indices.
        max_seq_length (int): Maximum sequence length for BERT input.

    Returns:
        inputs (dict): Dictionary containing tokenized and formatted inputs.
        labels (Tensor): Tensor containing label indices.
    """
    # Tokenize and format statements using the BERT tokenizer
    inputs = tokenizer(
        statements,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt"
    )
    labels = torch.tensor(statement_labels)
    return inputs, labels


def prepare_bert_model(train_labels, training_params, train_inputs, device):
    """
    Prepare a BERT model  by initializing the model, optimizer, and scheduler.

    Args:
        train_labels: Training labels.
        training_params (dict): Parameters for training the model.
        train_inputs: Tokenized and formatted training inputs.
        device: Device for training ("cuda" or "cpu").

    Returns:
        bert_model: Initialized BERT model.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        train_loader: Data loader for training.
    """
    # Initialize BERT model and optimizer
    bert_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(set(train_labels))
    ).to(device)
    optimizer = AdamW(
        bert_model.parameters(),
        lr=training_params['learning_rate']
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_inputs["input_ids"]) // (
            training_params['batch_size'] * training_params['epochs']
        )

    )

    # Prepare training data loader
    train_data = torch.utils.data.TensorDataset(
        train_inputs["input_ids"],
        train_inputs["attention_mask"],
        train_labels
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=training_params['batch_size'],
        shuffle=True
    )

    return bert_model, optimizer, scheduler, train_loader


def evaluate_bert_model(
        test_statements,
        test_statement_labels,
        training_params,
        device,
        bert_model,
        validation_labels,
        tokenizer,
        validation_inputs
        ):
    """
    Evaluate the BERT model's performance on the validation and test sets.

    Args:
        test_statements: Test statements.
        test_statement_labels: Test statement labels.
        training_params (dict): Parameters for training the model.
        device: Device for training ("cuda" or "cpu").
        bert_model: Trained BERT model.
        validation_labels: Validation labels.
        tokenizer: Pre-trained BERT tokenizer.
        validation_inputs: Tokenized and formatted validation inputs.
    """
    with torch.no_grad():
        # Move validation inputs to the specified device
        validation_inputs = {
            key: value.to(device) for key, value in validation_inputs.items()
        }

        # Get validation outputs from the BERT model
        validation_outputs = bert_model(**validation_inputs)
        validation_predictions = validation_outputs.logits.argmax(dim=1)

        # Calculate validation accuracy
        validation_accuracy = accuracy_score(
            validation_labels.to(device),
            validation_predictions.to(device)
        )

        # Generate and print validation classification report
        validation_classification_report = classification_report(
            validation_labels.to(device),
            validation_predictions.to(device)
        )
        print(f"Validation Accuracy: {validation_accuracy:.4f}")
        print("Validation Classification Report:")
        print(validation_classification_report)

        # Prepare test inputs for BERT model
        test_inputs, test_labels = prepare_bert_inputs(
            tokenizer,
            test_statements,
            test_statement_labels,
            training_params['max_seq_length']
        )

        # Move test inputs to the specified device
        test_inputs = {
            key: value.to(device) for key, value in test_inputs.items()
        }

        # Get test outputs from the BERT model
        test_outputs = bert_model(**test_inputs)
        test_predictions = test_outputs.logits.argmax(dim=1)

        # Calculate test accuracy
        test_accuracy = accuracy_score(
            test_labels.to(device),
            test_predictions.to(device)
        )

        # Generate and print test classification report
        test_classification_report = classification_report(
            test_labels.to(device),
            test_predictions.to(device)
        )
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("Test Classification Report:")
        print(test_classification_report)


def train_and_evaluate_bert_model(
        statements,
        statement_labels,
        validation_statements,
        validation_statement_labels,
        test_statements,
        test_statement_labels,
        training_params,
        device,
        save_path
        ):
    """
    Train and evaluate a BERT model using the specified data and parameters.

    Args:
        statements: Training statements.
        statement_labels: Training statement labels.
        validation_statements: Validation statements.
        validation_statement_labels: Validation statement labels.
        test_statements: Test statements.
        test_statement_labels: Test statement labels.
        training_params (dict): Parameters for training the model.
        device: Device for training ("cuda" or "cpu").
        save_path (str): Path to save the trained BERT model.

    Returns:
        bert_model: Trained BERT model.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_inputs, train_labels = prepare_bert_inputs(
        tokenizer,
        statements,
        statement_labels,
        training_params['max_seq_length']
    )
    validation_inputs, validation_labels = prepare_bert_inputs(
        tokenizer,
        validation_statements,
        validation_statement_labels,
        training_params['max_seq_length']
    )
    bert_model, optimizer, scheduler, train_loader = prepare_bert_model(
        train_labels,
        training_params,
        train_inputs,
        device
    )

    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(training_params['epochs']):
        bert_model.train()
        total_loss = 0

        print(f"Epoch {epoch + 1}/{training_params['epochs']} - Training...")
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids, attention_mask, batch_labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_labels = batch_labels.to(device)

            outputs = bert_model(
                input_ids,
                attention_mask=attention_mask,
                labels=batch_labels
            )
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            print(
                f"\rBatch {batch_idx + 1}/{len(train_loader)} "
                f"- Loss: "
                f"{loss.item():.4f}",
                end=""
            )

        average_loss = total_loss / len(train_loader)
        print(
            f"\nEpoch {epoch + 1}/{training_params['epochs']}, "
            f"Average Training Loss: {average_loss:.4f}"
        )

        # Evaluate on validation set
        bert_model.eval()
        evaluate_bert_model(
            test_statements,
            test_statement_labels,
            training_params,
            device,
            bert_model,
            validation_labels,
            tokenizer,
            validation_inputs
        )

    # Save the trained BERT model
    bert_model.save_pretrained(save_path)

    return bert_model

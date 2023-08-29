import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def create_tfidf_matrix(text_data, max_features=None):
    """
    Create a TF-IDF matrix from the provided text data.

    Args:
        text_data (list): List of text documents.
        max_features (int, optional): Maximum number of features for TF-IDF.

    Returns:
        tfidf_matrix (csr_matrix): TF-IDF matrix.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return tfidf_matrix, vectorizer


def evaluate_classification_model(model, X, y):
    """
    Evaluate the classification model's performance on the given data.

    Args:
        model: Trained classification model.
        X: Input features for evaluation.
        y: True labels for evaluation.

    Returns:
        accuracy (float): Accuracy score of the model.
        classification_report_str (str): Classification report as a string.
    """
    # Map label indices to their corresponding names
    labels = {
        0: "false",
        1: "half-true",
        2: "mostly-true",
        3: "true",
        4: "barely-true",
        5: "pants-fire"
    }
    # Make predictions using the model
    y_pred = model.predict(X)

    # Calculate accuracy score
    accuracy = accuracy_score(y, y_pred)

    # Generate classification report as a string
    classification_report_str = classification_report(
        y,
        y_pred,
        target_names=labels.values()
    )

    return accuracy, classification_report_str


def save_model(model, path):
    """
    Save the trained model to the specified path.

    Args:
        model: Trained model to be saved.
        path (str): Path to save the model.
    """
    torch.save(model, path)


def print_model_evaluation(model_name, accuracy, classification_report_str):
    """
    Print the evaluation results for a classification model.

    Args:
        model_name (str): Name of the model.
        accuracy (float): Accuracy score of the model.
        classification_report_str (str): Classification report as a string.
    """
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report_str)


def train_logistic_regression_model(
        X_train,
        y_train,
        train_style,
        training_params
        ):
    """
    Train a logistic regression model.

    Args:
        X_train: Training features.
        y_train: Training labels.
        train_style (str): Style of training ("normal", "grid", or "random").
        training_params (dict): Parameters for training the model.

    Returns:
        model: Trained logistic regression model.
        best_hyperparameters: Best hyperparameters if grid or random search.
    """
    if train_style == "normal":
        model = LogisticRegression(
            max_iter=1500,
            C=training_params['C'],
            solver=training_params['solver']
        )
    else:
        model = LogisticRegression(
            max_iter=1500,
            C=training_params['C'][0],
            solver=training_params['solver'][0]
        )

    if train_style == "grid":
        model = GridSearchCV(
            model,
            param_grid=training_params,
            cv=3
        )
    elif train_style == "random":
        model = RandomizedSearchCV(
            model,
            param_distributions=training_params,
            n_iter=10,
            cv=3
        )

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Determine best hyperparameters if applicable
    if train_style in ("grid", "random"):
        best_hyperparameters = model.best_params_
    else:
        best_hyperparameters = None

    return model, best_hyperparameters


def train_random_forest_model(X_train, y_train, train_style, training_params):
    """
    Train a random forest classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        train_style (str): Style of training ("normal", "grid", or "random").
        training_params (dict): Parameters for training the model.

    Returns:
        model: Trained random forest classifier.
        best_hyperparameters: Best hyperparameters if grid or random search.
    """
    if train_style == "normal":
        model = RandomForestClassifier(
            max_features=5000,
            n_estimators=training_params['n_estimators'],
            max_depth=training_params['max_depth'],
            min_samples_split=training_params['min_samples_split']
        )
    else:
        model = RandomForestClassifier(
            max_features=5000,
            n_estimators=training_params['n_estimators'][0],
            max_depth=training_params['max_depth'][0],
            min_samples_split=training_params['min_samples_split'][0]
        )

    if train_style == "grid":
        model = GridSearchCV(
            model,
            param_grid=training_params,
            cv=3
        )
    elif train_style == "random":
        model = RandomizedSearchCV(
            model,
            param_distributions=training_params,
            n_iter=10,
            cv=3
        )

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Determine best hyperparameters if applicable
    if train_style in ("grid", "random"):
        best_hyperparameters = model.best_params_
    else:
        best_hyperparameters = None

    return model, best_hyperparameters


def train_classification_model(
        algorithm,
        train_style,
        training_params,
        X_train,
        y_train
        ):
    """
    Train a classification model.

    Args:
        algorithm (str): The classification algorithm to
                         use ("logistic_regression" or "random_forest").
        train_style (str): The training style ("normal", "grid", or "random").
        training_params (dict): Parameters for training the model.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        model: Trained classification model.
        best_hyperparameters: Best hyperparameters if grid or random search.
    """
    if algorithm == "logistic_regression":
        return train_logistic_regression_model(
            X_train,
            y_train,
            train_style,
            training_params
        )
    elif algorithm == "random_forest":
        return train_random_forest_model(
            X_train,
            y_train,
            train_style,
            training_params
        )
    else:
        raise ValueError("Invalid algorithm name")


def evaluate_and_print_performance(
        model,
        X_validation,
        y_validation,
        X_test,
        y_test,
        algorithm_name
        ):
    """
    Evaluate the classification model's performance on the validation set.

    Args:
        model: Trained classification model.
        X_validation: Validation features.
        y_validation: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        algorithm_name (str): Name of the algorithm used.
    """
    if X_validation is not None and y_validation is not None:
        # Evaluate the model on the validation set
        validation_accuracy, validation_classification_report = \
            evaluate_classification_model(
                model,
                X_validation,
                y_validation
            )

        # Print the evaluation results
        print_model_evaluation(
            f"{algorithm_name} - Validation",
            validation_accuracy,
            validation_classification_report
        )

    # Evaluate the model on the test set
    test_accuracy, test_classification_report = evaluate_classification_model(
        model,
        X_test,
        y_test
    )

    # Print the evaluation results
    print_model_evaluation(
        f"{algorithm_name} - Test",
        test_accuracy,
        test_classification_report
    )


def train_and_evaluate_classification_model(
        algorithm,
        train_style,
        save_path,
        training_params,
        y_train,
        train_statements,
        y_validation,
        validation_statements,
        y_test,
        test_statements
        ):
    """
    Train and evaluate a classification model using the specified algorithm.

    Args:
        algorithm (str): The classification algorithm to
                         use ("logistic_regression" or "random_forest").
        train_style (str): The training style ("normal", "grid", or "random").
        save_path (str): Path to save the trained model.
        training_params (dict): Parameters for training the model.
        train_statements (list): List of training statement texts.
        train_statement_labels (list): List of training label indices.
        validation_statements (list): List of validation statement texts.
        validation_statement_labels (list): List of validation label indices.
        test_statements (list): List of test statement texts.
        test_statement_labels (list): List of test label indices.
    """
    X_train, vectorizer = create_tfidf_matrix(train_statements)
    X_test = vectorizer.transform(test_statements)
    X_validation = vectorizer.transform(validation_statements)

    if algorithm == "random_forest":
        X_validation = None
        y_validation = None

    if algorithm in ("logistic_regression", "random_forest"):
        # Train the specified classification model
        model, best_hyperparameters = train_classification_model(
            algorithm,
            train_style,
            training_params,
            X_train,
            y_train
        )

        # Save the trained model
        save_model(model, save_path)

        if train_style in ("grid", "random"):
            print("Best hyperparameters:", best_hyperparameters)

        # Evaluate and print performance on validation and test sets
        evaluate_and_print_performance(
            model,
            X_validation,
            y_validation,
            X_test,
            y_test,
            algorithm
        )

    else:
        raise ValueError("Invalid algorithm name")


def evaluate_and_print_model_performance(
        model,
        X_validation,
        y_validation,
        X_test,
        y_test,
        algorithm_name
        ):
    """
    Evaluate and print the performance of a trained model.

    Args:
        model: Trained classification model.
        X_validation: Validation features.
        y_validation: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        algorithm_name (str): Name of the algorithm used.
    """
    if X_validation is not None and y_validation is not None:
        # Evaluate and print performance on validation set
        validation_accuracy, validation_classification_report = \
            evaluate_classification_model(
                model,
                X_validation,
                y_validation
            )
        print_model_evaluation(f"{algorithm_name} - Validation",
                               validation_accuracy,
                               validation_classification_report
                               )

    # Evaluate and print performance on test set
    test_accuracy, test_classification_report = evaluate_classification_model(
        model,
        X_test,
        y_test
    )
    print_model_evaluation(
        f"{algorithm_name} - Test",
        test_accuracy,
        test_classification_report
    )

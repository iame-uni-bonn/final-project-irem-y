"""
Function to train a model to recognize fake news with random forest algorithm.

Functions:
    - vectorize_text_and_other_features(
        df,
        text_features,
        other_features,
        max_features
        )
    - train_random_forest_classifier(
        X_train,
        y_train,
        random_state,
        n_estimators,
        max_depth,
        min_samples_leaf,
        min_samples_split
        )
    - train_random_forest_classifier_grid(
        X_train,
        y_train,
        random_state,
        param_grid
        )
    - train_random_forest_classifier_random(
        X_train,
        y_train,
        random_state,
        param_dist
        )
    - save_random_forest_classifier_model(model, path)
    - evaluate_random_forest_classifier_performance(
        model,
        test_df,
        text_features,
        other_features,
        max_features
        )
    - train_and_evaluate_random_forest_classifier(
        save_path,
        train_type,
        max_features,
        random_state,
        trainings_parameter
        )
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
from general_functions import load_liar_dataset, prepare_dataframe


def vectorize_text_and_other_features(
        df,
        text_features,
        other_features,
        max_features
        ):
    """
    Vectorize text and other features using TF-IDF and combine them.

    Args:
        df (DataFrame): The input DataFrame.
        text_features (list): List of columns containing text features.
        other_features (list): List of columns containing non-text features.
        max_features (int): Maximum number of features for TF-IDF.

    Returns:
        csr_matrix: Combined feature matrix.
    """
    # Use TF-IDF vectorization for text features
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_text_features = vectorizer.fit_transform(
        df[text_features].astype(str).values.sum(axis=1)
    )
    X_other_features = df[other_features].values

    # Combine text and other features
    return hstack((X_text_features, X_other_features))


def train_random_forest_classifier(
        X_train,
        y_train,
        random_state,
        n_estimators,
        max_depth,
        min_samples_leaf,
        min_samples_split
        ):
    """
    Train a Random Forest classifier.

    Args:
        X_train (csr_matrix): Training feature matrix.
        y_train (array-like): Training labels.
        random_state (int): Seed for random number generation.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of each tree.
        min_samples_leaf (int): Minimum number of samples in leaf nodes.
        min_samples_split (int): Minimum number of samples required to split
        an internal node.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
    """
    try:
        # Create a Random Forest model
        random_forest_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            n_jobs=-1
        )

        # Train the model
        random_forest_model.fit(X_train, y_train)

        # Log messages
        print("Training completed.")

        return random_forest_model
    except Exception as e:
        print("Error during model training:", str(e))
        raise


def train_random_forest_classifier_grid(
        X_train,
        y_train,
        random_state,
        param_grid
        ):
    """
    Train a Random Forest classifier using grid search.

    Args:
        X_train (csr_matrix): Training feature matrix.
        y_train (array-like): Training labels.
        random_state (int): Seed for random number generation.
        param_grid (dict): Dictionary specifying the hyperparameter grid to
        search.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
    """
    try:
        # Create a Random Forest model
        random_forest_model = RandomForestClassifier(
            random_state=random_state
        )

        # Perform grid search
        grid_search = GridSearchCV(
            random_forest_model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1
        )

        # Fit the model with grid search
        grid_search.fit(X_train, y_train)

        # Get the best parameters
        best_params = grid_search.best_params_
        print("Best Parameters:", best_params)

        # Get the best model from grid search
        random_forest_model = grid_search.best_estimator_

        # Log messages
        print("Grid search training completed.")

        return random_forest_model

    except Exception as e:
        print("Error during model training:", str(e))
        raise


def train_random_forest_classifier_random(
        X_train,
        y_train,
        random_state,
        param_dist
        ):
    """
    Train a Random Forest classifier using random search.

    Args:
        X_train (csr_matrix): Training feature matrix.
        y_train (array-like): Training labels.
        random_state (int): Seed for random number generation.
        param_dist (dict): Dictionary specifying the hyperparameter
        distributions for random search.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
    """
    try:

        # Create a Random Forest model
        random_forest_model = RandomForestClassifier(
            random_state=random_state
        )

        # Perform random search
        random_search = RandomizedSearchCV(
            random_forest_model,
            param_distributions=param_dist,
            n_iter=10,
            cv=5,
            random_state=random_state,
            n_jobs=-1
        )

        # Fit the model with random search
        random_search.fit(X_train, y_train)

        # Get the best parameters
        best_params = random_search.best_params_
        print("Best Parameters:", best_params)

        # Get the best model from random search
        random_forest_model = random_search.best_estimator_

        # Log messages
        print("Random search training completed.")

        return random_forest_model

    except Exception as e:
        print("Error during model training:", str(e))
        raise


def save_random_forest_classifier_model(model, path):
    """
    Save a Random Forest classifier model to a file.

    Args:
        model (RandomForestClassifier): Trained Random Forest model.
        path (str): Path to save the model.

    Returns:
        None
    """
    try:
        # Save the Random Forest model using joblib
        with open(path, 'wb') as model_file:
            joblib.dump(model, model_file)

        # Log messages
        print(f"Model saved to {path}")
    except Exception as e:
        print("Error saving the model:", str(e))
        raise


def evaluate_random_forest_classifier_performance(
        model,
        test_df,
        text_features,
        other_features,
        max_features
        ):
    """
    Evaluate the performance of a Random Forest classifier.

    Args:
        model (RandomForestClassifier): Trained Random Forest model.
        test_df (DataFrame): Test DataFrame.
        text_features (list): List of columns containing text features.
        other_features (list): List of columns containing non-text features.

    Returns:
        tuple: A tuple containing accuracy and classification report.
    """
    X_test = vectorize_text_and_other_features(
        test_df,
        text_features,
        other_features,
        max_features
    )
    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(test_df['label'], predictions)
    classification_rep = classification_report(test_df['label'], predictions)

    return accuracy, classification_rep


def train_and_evaluate_random_forest_classifier(
        save_path,
        train_type,
        max_features,
        random_state,
        trainings_parameter
        ):
    """
    Train and evaluate a Random Forest classifier.

    Args:
        save_path (str): Path to save the trained model.
        train_type (str): Style of training ("normal", "grid", or "random").
        max_features (int): Maximum number of features for TF-IDF.
        random_state (int): Seed for random number generation.
        trainings_parameter (dict): All needed trainings parameters, single or
        as grid.

    Returns:
        None
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

    # Vectorize the data
    X_train = vectorize_text_and_other_features(
        train_df,
        text_feature_columns,
        other_feature_columns,
        max_features
    )

    # Extract the label column as the target
    Y_train = train_df['label']

    # Train the model
    if train_type == "basic":
        random_forest_model = train_random_forest_classifier(
            X_train=X_train,
            y_train=Y_train,
            random_state=random_state,
            n_estimators=trainings_parameter["n_estimators"],
            max_depth=trainings_parameter["max_depth"],
            min_samples_leaf=trainings_parameter["min_samples_leaf"],
            min_samples_split=trainings_parameter["min_samples_split"]
        )
    elif train_type == "grid":
        random_forest_model = train_random_forest_classifier_grid(
            X_train=X_train,
            y_train=Y_train,
            random_state=random_state,
            param_grid=trainings_parameter
        )
    elif train_type == "random":
        random_forest_model = train_random_forest_classifier_random(
            X_train=X_train,
            y_train=Y_train,
            random_state=random_state,
            param_dist=trainings_parameter
        )

    # Save the model
    save_random_forest_classifier_model(random_forest_model, save_path)

    # Evaluate the trained model
    accuracy, classification_rep =\
        evaluate_random_forest_classifier_performance(
            random_forest_model,
            test_df,
            text_feature_columns,
            other_feature_columns,
            max_features
        )

    # Log evaluation results
    print("Evaluation results:")
    print("Accuracy: ", accuracy)
    print("Classification Report:\n", classification_rep)

    return accuracy, classification_rep

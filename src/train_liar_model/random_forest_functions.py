from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
import joblib
from main_functions import load_liar_dataset, prepare_dataframe


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
        n_estimators,
        random_state
        ):
    """
    Train a Random Forest classifier.

    Args:
        X_train (csr_matrix): Training feature matrix.
        y_train (array-like): Training labels.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Seed for random number generation.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
    """
    try:
        # Create a Random Forest model
        random_forest_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

        # Train the model
        random_forest_model.fit(X_train, y_train)

        # Log messages
        print("Training completed.")

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

    # Log evaluation results
    print("Evaluation results:")
    print("Accuracy: ", accuracy)
    print("Classification Report:\n", classification_rep)


def train_and_evaluate_random_forest_classifier(
        save_path,
        max_features,
        n_estimators,
        random_state
        ):
    """
    Train and evaluate a Random Forest classifier.

    Args:
        save_path (str): Path to save the trained model.
        max_features (int): Maximum number of features for TF-IDF.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Seed for random number generation.
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
    random_forest_model = train_random_forest_classifier(
        X_train,
        Y_train,
        n_estimators,
        random_state
    )

    # Save the model
    save_random_forest_classifier_model(random_forest_model, save_path)

    # Evaluate the trained model
    evaluate_random_forest_classifier_performance(
        random_forest_model,
        test_df,
        text_feature_columns,
        other_feature_columns,
        max_features
    )

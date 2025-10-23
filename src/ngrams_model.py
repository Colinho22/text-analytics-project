# N-Grams baseline model for text classification.
# This module implements traditional count-based features (TF-IDF) with
# classical machine learning classifiers (Naive Bayes and Logistic Regression).

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


class NGramClassifier:

    def __init__(
            self,
            classifier_type: str = 'logistic',
            ngram_range: Tuple[int, int] = (1, 3),
            max_features: int = 10000,
            min_df: int = 2,
            max_df: float = 0.95,
            use_idf: bool = True,
            random_state: int = 42
    ):

        self.classifier_type = classifier_type
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.random_state = random_state

        # initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            use_idf=use_idf,
            sublinear_tf=True,
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )

        # initialize classifier
        if classifier_type == 'naive_bayes':
            self.classifier = MultinomialNB()
        elif classifier_type == 'logistic':
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                solver='saga',
                n_jobs=-1,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        # store class names
        self.classes_ = None


    def fit(self, texts: List[str], labels: np.ndarray) -> 'NGramClassifier':
        print(f"\nTraining {self.classifier_type.upper()} classifier...")
        print(f"N-gram range: {self.ngram_range}")
        print(f"Max features: {self.max_features}")

        # fit vectorizer and transform texts
        X = self.vectorizer.fit_transform(texts)
        print(f"Feature matrix shape: {X.shape}")

        # store class names
        self.classes_ = np.unique(labels)

        # fit classifier
        self.classifier.fit(X, labels)

        print(f"✅ Training complete!")

        return self


    def predict(self, texts: List[str]) -> np.ndarray:
        # transform texts
        X = self.vectorizer.transform(texts)

        # predict
        predictions = self.classifier.predict(X)

        return predictions


    def predict_proba(self, texts: List[str]) -> np.ndarray:
        # transform texts
        X = self.vectorizer.transform(texts)

        # predict probabilities
        probabilities = self.classifier.predict_proba(X)

        return probabilities


    def get_top_features(
            self,
            class_name: str,
            top_n: int = 20
    ) -> List[Tuple[str, float]]:
        if self.classifier_type != 'logistic':
            print("⚠️  Feature extraction only supported for logistic regression")
            return []

        # find class index
        try:
            class_idx = list(self.classes_).index(class_name)
        except ValueError:
            print(f"⚠️  Class '{class_name}' not found")
            return []

        # get feature names
        feature_names = self.vectorizer.get_feature_names_out()

        # get weights for this class
        weights = self.classifier.coef_[class_idx]

        # get top features by absolute weight
        top_indices = np.argsort(np.abs(weights))[-top_n:][::-1]

        top_features = [
            (feature_names[i], float(weights[i]))
            for i in top_indices
        ]

        return top_features


# get top features for classes
    def get_all_top_features(
            self,
            top_n: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        all_features = {}

        for class_name in self.classes_:
            all_features[class_name] = self.get_top_features(class_name, top_n)

        return all_features

    def save(self, filepath: str) -> None:
        # create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # save model
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'classes': self.classes_,
            'classifier_type': self.classifier_type,
            'ngram_range': self.ngram_range,
            'max_features': self.max_features
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Saved model to {filepath}")

# load models
    @classmethod
    def load(cls, filepath: str) -> 'NGramClassifier':

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # create instance
        instance = cls(classifier_type=model_data['classifier_type'])

        # restore components
        instance.vectorizer = model_data['vectorizer']
        instance.classifier = model_data['classifier']
        instance.classes_ = model_data['classes']
        instance.ngram_range = model_data['ngram_range']
        instance.max_features = model_data['max_features']

        print(f"✅ Loaded model from {filepath}")

        return instance


def train_both_classifiers(
        train_texts: List[str],
        train_labels: np.ndarray,
        ngram_range: Tuple[int, int] = (1, 3),
        max_features: int = 10000
) -> Dict[str, NGramClassifier]:
    # train both naive bayes and logistic regression classifiers
    models = {}

    print("\n" + "=" * 60)
    print("TRAINING N-GRAM MODELS")
    print("=" * 60)

    # Train Naive Bayes
    print("\n1. NAIVE BAYES")
    print("-" * 60)
    nb_model = NGramClassifier(
        classifier_type='naive_bayes',
        ngram_range=ngram_range,
        max_features=max_features
    )
    nb_model.fit(train_texts, train_labels)
    models['naive_bayes'] = nb_model

    # train Logistic Regression
    print("\n2. LOGISTIC REGRESSION")
    print("-" * 60)
    lr_model = NGramClassifier(
        classifier_type='logistic',
        ngram_range=ngram_range,
        max_features=max_features
    )
    lr_model.fit(train_texts, train_labels)
    models['logistic'] = lr_model

    print("\n" + "=" * 60)
    print("✅ Both models trained!")
    print("=" * 60)

    return models


if __name__ == "__main__":
    # test the N-Gram classifier
    print("N-Gram Classifier - Test Run")
    print("=" * 60)

    # create synthetic test data
    np.random.seed(42)

    train_texts = [
                      "machine learning artificial intelligence deep neural networks",
                      "stock market trading investment financial analysis",
                      "football soccer basketball sports competition",
                      "machine learning models prediction algorithms",
                      "stock prices market analysis trading strategies",
                      "sports team championship victory competition"
                  ] * 100

    train_labels = np.array(
        ['Technology', 'Business', 'Sports'] * 2 * 100
    )

    test_texts = [
        "deep learning neural network training",
        "financial market investment portfolio",
        "basketball game championship final"
    ]

    print(f"\nTraining data: {len(train_texts)} samples")
    print(f"Test data: {len(test_texts)} samples")
    print(f"Classes: {np.unique(train_labels)}")

    # train Logistic Regression model
    model = NGramClassifier(
        classifier_type='logistic',
        ngram_range=(1, 2),
        max_features=1000
    )

    model.fit(train_texts, train_labels)

    # test predictions
    print("\n" + "=" * 60)
    print("TEST PREDICTIONS")
    print("=" * 60)

    predictions = model.predict(test_texts)
    probabilities = model.predict_proba(test_texts)

    for text, pred, probs in zip(test_texts, predictions, probabilities):
        print(f"\nText: {text}")
        print(f"Prediction: {pred}")
        print(f"Confidence: {probs.max():.3f}")

    # show top features
    print("\n" + "=" * 60)
    print("TOP FEATURES PER CLASS")
    print("=" * 60)

    for class_name in model.classes_:
        top_features = model.get_top_features(class_name, top_n=5)
        print(f"\n{class_name}:")
        for feature, weight in top_features:
            print(f"  {feature:20s}: {weight:6.3f}")

    # test save/load
    print("\n" + "=" * 60)
    print("TESTING SAVE/LOAD")
    print("=" * 60)

    # auto-detect project root
    current_file = Path(__file__).resolve()
    if current_file.parent.name == 'src':
        project_root = current_file.parent.parent
    else:
        project_root = current_file.parent

    model_path = project_root / 'models' / 'test_ngram_model.pkl'
    model.save(str(model_path))

    # load and test
    loaded_model = NGramClassifier.load(str(model_path))
    loaded_predictions = loaded_model.predict(test_texts)

    print(f"\nOriginal predictions: {predictions}")
    print(f"Loaded predictions:   {loaded_predictions}")
    print(f"Match: {np.array_equal(predictions, loaded_predictions)}")

    print("\n" + "=" * 60)
    print("✅ N-Gram classifier test complete!")
    print("=" * 60)
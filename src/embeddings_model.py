# Word2Vec Embeddings model for topic classification.
# This module trains Word2Vec on the dataset to learn semantic word representations,
# then averages word vectors to create document embeddings for classification.

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


class EmbeddingsClassifier:

    def __init__(
            self,
            classifier_type: str = 'logistic',
            vector_size: int = 100,
            window: int = 5,
            min_count: int = 2,
            epochs: int = 20,
            workers: int = -1,
            use_tfidf: bool = True,
            random_state: int = 42
    ):

        self.classifier_type = classifier_type
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.workers = workers
        self.use_tfidf = use_tfidf
        self.random_state = random_state

        # initialize Word2Vec model
        self.word2vec_model = None

        # initialize TF-IDF vectorizer for weighted averaging
        self.tfidf_vectorizer = None
        self.idf_dict = None

        # initialize classifier
        if classifier_type == 'logistic':
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                solver='saga',
                n_jobs=-1,
                class_weight='balanced'
            )
        elif classifier_type == 'svm':
            self.classifier = SVC(
                kernel='rbf',
                random_state=random_state,
                probability=True,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        # store class names
        self.classes_ = None

    def _tokenize_texts(self, texts: List[str]) -> List[List[str]]:
        # tokenize texts into list of words for Word2Vec training
        tokenized = []
        for text in texts:
            # simple whitespace tokenization
            tokens = text.lower().split()
            tokenized.append(tokens)
        return tokenized

    def _get_document_embedding(self, tokens: List[str], text: str = None) -> np.ndarray:
        # create document embedding by averaging word vectors
        # optionally weight by TF-IDF scores for better representation

        if not self.use_tfidf or self.idf_dict is None:
            # simple averaging (original approach)
            word_vectors = []

            for token in tokens:
                if token in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[token])

            if len(word_vectors) == 0:
                # if no words found, return zero vector
                return np.zeros(self.vector_size)

            # average all word vectors
            doc_vector = np.mean(word_vectors, axis=0)

            return doc_vector

        else:
            # TF-IDF weighted averaging
            # calculate TF (term frequency) for this document
            token_counts = {}
            for token in tokens:
                if token in self.word2vec_model.wv:
                    token_counts[token] = token_counts.get(token, 0) + 1

            if len(token_counts) == 0:
                return np.zeros(self.vector_size)

            # calculate TF-IDF weights
            weighted_vectors = []
            weights = []

            total_tokens = len(tokens)

            for token, count in token_counts.items():
                if token in self.word2vec_model.wv:
                    # TF: term frequency in document
                    tf = count / total_tokens

                    # IDF: inverse document frequency (from fitted vectorizer)
                    idf = self.idf_dict.get(token, 1.0)

                    # TF-IDF weight
                    tfidf_weight = tf * idf

                    # weight the word vector
                    weighted_vectors.append(self.word2vec_model.wv[token] * tfidf_weight)
                    weights.append(tfidf_weight)

            if len(weighted_vectors) == 0:
                return np.zeros(self.vector_size)

            # weighted average
            doc_vector = np.sum(weighted_vectors, axis=0) / (np.sum(weights) + 1e-10)

            return doc_vector

    def fit(self, texts: List[str], labels: np.ndarray) -> 'EmbeddingsClassifier':
        print(f"\nTraining Word2Vec Embeddings + {self.classifier_type.upper()} classifier...")
        print(f"Vector size: {self.vector_size}")
        print(f"Window: {self.window}")
        print(f"Min count: {self.min_count}")
        print(f"Epochs: {self.epochs}")
        print(f"TF-IDF weighting: {self.use_tfidf}")

        # tokenize texts
        print("\nTokenizing texts...")
        tokenized_texts = self._tokenize_texts(texts)

        # train Word2Vec model
        print("Training Word2Vec model...")
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            seed=self.random_state
        )

        print(f"Vocabulary size: {len(self.word2vec_model.wv):,} words")

        # train TF-IDF vectorizer if enabled
        if self.use_tfidf:
            print("Fitting TF-IDF vectorizer for weighted averaging...")
            self.tfidf_vectorizer = TfidfVectorizer(
                vocabulary=set(self.word2vec_model.wv.index_to_key),
                lowercase=True,
                token_pattern=r'\b\w+\b'
            )

            # fit on original texts
            self.tfidf_vectorizer.fit(texts)

            # extract IDF weights
            self.idf_dict = dict(zip(
                self.tfidf_vectorizer.get_feature_names_out(),
                self.tfidf_vectorizer.idf_
            ))

            print(f"TF-IDF vocabulary: {len(self.idf_dict):,} words")

        # create document embeddings
        print("Creating document embeddings...")
        X = np.array([
            self._get_document_embedding(tokens, text)
            for tokens, text in zip(tokenized_texts, texts)
        ])

        print(f"Document embeddings shape: {X.shape}")

        # store class names
        self.classes_ = np.unique(labels)

        # fit classifier
        print(f"Training {self.classifier_type} classifier...")
        self.classifier.fit(X, labels)

        print(f"✅ Training complete!")

        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        # tokenize texts
        tokenized_texts = self._tokenize_texts(texts)

        # create document embeddings
        X = np.array([
            self._get_document_embedding(tokens, text)
            for tokens, text in zip(tokenized_texts, texts)
        ])

        # predict
        predictions = self.classifier.predict(X)

        return predictions

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        # tokenize texts
        tokenized_texts = self._tokenize_texts(texts)

        # create document embeddings
        X = np.array([
            self._get_document_embedding(tokens, text)
            for tokens, text in zip(tokenized_texts, texts)
        ])

        # predict probabilities
        probabilities = self.classifier.predict_proba(X)

        return probabilities

    def get_word_embedding(self, word: str) -> Optional[np.ndarray]:
        # get embedding vector for a single word
        word = word.lower()

        if word in self.word2vec_model.wv:
            return self.word2vec_model.wv[word]
        else:
            return None

    def find_similar_words(
            self,
            word: str,
            top_n: int = 10
    ) -> List[Tuple[str, float]]:
        # find most similar words to a given word
        word = word.lower()

        if word not in self.word2vec_model.wv:
            print(f"⚠️  Word '{word}' not in vocabulary")
            return []

        similar = self.word2vec_model.wv.most_similar(word, topn=top_n)

        return similar

    def get_category_words(
            self,
            texts: List[str],
            labels: np.ndarray,
            class_name: str,
            top_n: int = 20
    ) -> List[Tuple[str, float]]:
        # get most frequent/representative words for a category
        # filters out stop words and numeric-only tokens

        # filter texts for this class
        class_texts = [
            text for text, label in zip(texts, labels)
            if label == class_name
        ]

        # tokenize and count words
        word_counts = {}
        for text in class_texts:
            tokens = text.lower().split()
            for token in tokens:
                # skip stop words, numeric-only tokens, and only count words in vocabulary
                if (token in self.word2vec_model.wv and
                        token not in ENGLISH_STOP_WORDS and
                        not token.isdigit()):
                    word_counts[token] = word_counts.get(token, 0) + 1

        # sort by frequency
        top_words = sorted(
            word_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        return top_words

    def save(self, filepath: str) -> None:
        # create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # save model
        model_data = {
            'word2vec_model': self.word2vec_model,
            'classifier': self.classifier,
            'classes': self.classes_,
            'classifier_type': self.classifier_type,
            'vector_size': self.vector_size,
            'window': self.window,
            'min_count': self.min_count,
            'use_tfidf': self.use_tfidf,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'idf_dict': self.idf_dict
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Saved model to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'EmbeddingsClassifier':

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # create instance
        instance = cls(
            classifier_type=model_data['classifier_type'],
            use_tfidf=model_data.get('use_tfidf', False)
        )

        # restore components
        instance.word2vec_model = model_data['word2vec_model']
        instance.classifier = model_data['classifier']
        instance.classes_ = model_data['classes']
        instance.vector_size = model_data['vector_size']
        instance.window = model_data['window']
        instance.min_count = model_data['min_count']
        instance.tfidf_vectorizer = model_data.get('tfidf_vectorizer', None)
        instance.idf_dict = model_data.get('idf_dict', None)

        print(f"✅ Loaded model from {filepath}")

        return instance


def train_both_classifiers(
        train_texts: List[str],
        train_labels: np.ndarray,
        vector_size: int = 100,
        epochs: int = 20,
        use_tfidf: bool = True
) -> Dict[str, EmbeddingsClassifier]:
    # train both logistic regression and SVM classifiers
    models = {}

    print("\n" + "=" * 60)
    print("TRAINING EMBEDDINGS MODELS")
    print("=" * 60)

    # train Logistic Regression
    print("\n1. LOGISTIC REGRESSION")
    print("-" * 60)
    lr_model = EmbeddingsClassifier(
        classifier_type='logistic',
        vector_size=vector_size,
        epochs=epochs,
        use_tfidf=use_tfidf
    )
    lr_model.fit(train_texts, train_labels)
    models['logistic'] = lr_model

    # train SVM
    print("\n2. SVM")
    print("-" * 60)
    svm_model = EmbeddingsClassifier(
        classifier_type='svm',
        vector_size=vector_size,
        epochs=epochs,
        use_tfidf=use_tfidf
    )
    svm_model.fit(train_texts, train_labels)
    models['svm'] = svm_model

    print("\n" + "=" * 60)
    print("✅ Both models trained!")
    print("=" * 60)

    return models


if __name__ == "__main__":
    # test the Embeddings classifier
    print("Embeddings Classifier - Test Run")
    print("=" * 60)

    # create synthetic test data
    np.random.seed(42)

    train_texts = [
                      "machine learning artificial intelligence deep neural networks algorithms",
                      "stock market trading investment financial analysis portfolio",
                      "football soccer basketball sports competition championship",
                      "python programming software development code algorithms",
                      "business finance company revenue profit investment",
                      "athlete game team victory sports competition"
                  ] * 100

    train_labels = np.array(
        ['Technology', 'Business', 'Sports'] * 2 * 100
    )

    test_texts = [
        "deep learning neural network training algorithms",
        "financial market investment portfolio analysis",
        "basketball game championship final victory"
    ]

    print(f"\nTraining data: {len(train_texts)} samples")
    print(f"Test data: {len(test_texts)} samples")
    print(f"Classes: {np.unique(train_labels)}")

    # train Logistic Regression model
    model = EmbeddingsClassifier(
        classifier_type='logistic',
        vector_size=50,
        epochs=10,
        use_tfidf=True
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

    # test word similarity
    print("\n" + "=" * 60)
    print("WORD SIMILARITY")
    print("=" * 60)

    test_words = ['machine', 'stock', 'football']
    for word in test_words:
        print(f"\nWords similar to '{word}':")
        similar = model.find_similar_words(word, top_n=5)
        for similar_word, score in similar:
            print(f"  {similar_word:20s}: {score:.3f}")

    # test category words
    print("\n" + "=" * 60)
    print("TOP CATEGORY WORDS")
    print("=" * 60)

    for class_name in model.classes_:
        top_words = model.get_category_words(
            train_texts,
            train_labels,
            class_name,
            top_n=5
        )
        print(f"\n{class_name}:")
        for word, count in top_words:
            print(f"  {word:20s}: {count} occurrences")

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

    model_path = project_root / 'models' / 'test_embeddings_model.pkl'
    model.save(str(model_path))

    # load and test
    loaded_model = EmbeddingsClassifier.load(str(model_path))
    loaded_predictions = loaded_model.predict(test_texts)

    print(f"\nOriginal predictions: {predictions}")
    print(f"Loaded predictions:   {loaded_predictions}")
    print(f"Match: {np.array_equal(predictions, loaded_predictions)}")

    print("\n" + "=" * 60)
    print("✅ Embeddings classifier test complete!")
    print("=" * 60)
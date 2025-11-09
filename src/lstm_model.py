# LSTM model for topic classification.
# This module implements a Bidirectional LSTM with embeddings trained from scratch.
# The model learns both word representations and sequential patterns end-to-end.

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import pickle

# use tensorflow/keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Bidirectional, Dense, Dropout, Input
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class LSTMClassifier:

    def __init__(
            self,
            vocab_size: int = 20000,
            embedding_dim: int = 200,
            max_sequence_length: int = 250,
            lstm_units: int = 128,
            num_layers: int = 2,
            dropout: float = 0.4,
            recurrent_dropout: float = 0.2,
            random_state: int = 42
    ):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.random_state = random_state

        # tokenizer for text to sequences
        self.tokenizer = None

        # keras model
        self.model = None

        # label encoder
        self.label_encoder = None
        self.classes_ = None
        self.num_classes = None

        # training history
        self.history = None

    def _build_model(self):
        # build bidirectional LSTM architecture

        model = Sequential()

        # embedding layer (trained from scratch)
        model.add(Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_length,
            name='embedding'
        ))

        # bidirectional LSTM layers
        for i in range(self.num_layers):
            return_sequences = (i < self.num_layers - 1)

            model.add(Bidirectional(
                LSTM(
                    self.lstm_units,
                    return_sequences=return_sequences,
                    dropout=self.dropout,
                    recurrent_dropout=self.recurrent_dropout
                ),
                name=f'bilstm_{i + 1}'
            ))

        # dropout for regularization
        model.add(Dropout(self.dropout, name='dropout'))

        # output layer
        model.add(Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        ))

        # compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def fit(
            self,
            texts: List[str],
            labels: np.ndarray,
            validation_data: Optional[Tuple[List[str], np.ndarray]] = None,
            batch_size: int = 32,
            epochs: int = 20,
            early_stopping_patience: int = 3,
            verbose: int = 1
    ) -> 'LSTMClassifier':

        print(f"\nTraining LSTM classifier...")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Embedding dim: {self.embedding_dim}")
        print(f"Max sequence length: {self.max_sequence_length}")
        print(f"LSTM units: {self.lstm_units}")
        print(f"Num layers: {self.num_layers}")
        print(f"Dropout: {self.dropout}")

        # fit tokenizer on training texts
        print("\nFitting tokenizer...")
        self.tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token='<OOV>',
            lower=True
        )
        self.tokenizer.fit_on_texts(texts)

        actual_vocab_size = min(len(self.tokenizer.word_index) + 1, self.vocab_size)
        print(f"Actual vocabulary size: {actual_vocab_size}")

        # convert texts to sequences
        print("Converting texts to sequences...")
        X_train = self.tokenizer.texts_to_sequences(texts)
        X_train = pad_sequences(
            X_train,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )

        # encode labels
        self.classes_ = np.unique(labels)
        self.num_classes = len(self.classes_)
        self.label_encoder = {label: idx for idx, label in enumerate(self.classes_)}

        y_train_encoded = np.array([self.label_encoder[label] for label in labels])
        y_train = to_categorical(y_train_encoded, num_classes=self.num_classes)

        print(f"\nTraining data shape: {X_train.shape}")
        print(f"Labels shape: {y_train.shape}")
        print(f"Number of classes: {self.num_classes}")

        # prepare validation data if provided
        validation_split = None
        val_data = None

        if validation_data is not None:
            val_texts, val_labels = validation_data

            X_val = self.tokenizer.texts_to_sequences(val_texts)
            X_val = pad_sequences(
                X_val,
                maxlen=self.max_sequence_length,
                padding='post',
                truncating='post'
            )

            y_val_encoded = np.array([self.label_encoder[label] for label in val_labels])
            y_val = to_categorical(y_val_encoded, num_classes=self.num_classes)

            val_data = (X_val, y_val)
            print(f"Validation data shape: {X_val.shape}")
        else:
            validation_split = 0.1

        # build model
        print("\nBuilding model architecture...")
        self.model = self._build_model()

        print("\nModel summary:")
        self.model.summary()

        # callbacks
        callbacks = []

        if validation_data is not None or validation_split is not None:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)

        # train model
        print(f"\nTraining model for up to {epochs} epochs...")
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=val_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )

        # store training history
        self.history = history.history

        print(f"\n✅ Training complete!")

        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        # convert texts to sequences
        X = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(
            X,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )

        # predict
        y_pred_proba = self.model.predict(X, verbose=0)
        y_pred_encoded = np.argmax(y_pred_proba, axis=1)

        # decode labels
        label_decoder = {idx: label for label, idx in self.label_encoder.items()}
        predictions = np.array([label_decoder[idx] for idx in y_pred_encoded])

        return predictions

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        # convert texts to sequences
        X = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(
            X,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )

        # predict probabilities
        probabilities = self.model.predict(X, verbose=0)

        return probabilities

    def get_training_history(self) -> dict:
        # return training history as a dictionary
        if self.history is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self.history

    def save(self, filepath: str) -> None:
        # create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # save model weights
        model_path = filepath.replace('.pkl', '_model.keras')
        self.model.save(model_path)

        # save tokenizer and metadata
        metadata = {
            'tokenizer': self.tokenizer,
            'label_encoder': self.label_encoder,
            'classes': self.classes_,
            'num_classes': self.num_classes,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'max_sequence_length': self.max_sequence_length,
            'lstm_units': self.lstm_units,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'history': self.history
        }

        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"✅ Saved model to {model_path}")
        print(f"✅ Saved metadata to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'LSTMClassifier':
        # load metadata
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)

        # create instance
        instance = cls(
            vocab_size=metadata['vocab_size'],
            embedding_dim=metadata['embedding_dim'],
            max_sequence_length=metadata['max_sequence_length'],
            lstm_units=metadata['lstm_units'],
            num_layers=metadata['num_layers'],
            dropout=metadata['dropout'],
            recurrent_dropout=metadata.get('recurrent_dropout', 0.2)
        )

        # restore metadata
        instance.tokenizer = metadata['tokenizer']
        instance.label_encoder = metadata['label_encoder']
        instance.classes_ = metadata['classes']
        instance.num_classes = metadata['num_classes']
        instance.history = metadata.get('history', None)

        # load model weights
        model_path = filepath.replace('.pkl', '_model.keras')
        instance.model = keras.models.load_model(model_path)

        print(f"✅ Loaded model from {model_path}")
        print(f"✅ Loaded metadata from {filepath}")

        return instance


if __name__ == "__main__":
    # test the LSTM classifier
    print("LSTM Classifier - Test Run")
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

    # train LSTM model
    model = LSTMClassifier(
        vocab_size=1000,
        embedding_dim=50,
        max_sequence_length=20,
        lstm_units=32,
        num_layers=1,
        dropout=0.3
    )

    model.fit(
        train_texts,
        train_labels,
        batch_size=16,
        epochs=5,
        verbose=1
    )

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

    model_path = project_root / 'models' / 'test_lstm_model.pkl'
    model.save(str(model_path))

    # load and test
    loaded_model = LSTMClassifier.load(str(model_path))
    loaded_predictions = loaded_model.predict(test_texts)

    print(f"\nOriginal predictions: {predictions}")
    print(f"Loaded predictions:   {loaded_predictions}")
    print(f"Match: {np.array_equal(predictions, loaded_predictions)}")

    print("\n" + "=" * 60)
    print("✅ LSTM classifier test complete!")
    print("=" * 60)
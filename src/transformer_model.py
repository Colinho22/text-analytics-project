# Transformer model for topic classification.
# This module implements DistilBERT fine-tuning for text classification.
# Uses pre-trained contextual embeddings and self-attention mechanism.

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import pickle
import warnings

# suppress warnings
warnings.filterwarnings('ignore')

# transformers library
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class TransformerClassifier:

    def __init__(
            self,
            model_name: str = 'distilbert-base-uncased',
            max_length: int = 512,
            learning_rate: float = 2e-5,
            warmup_steps: int = 500,
            random_state: int = 42
    ):

        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.random_state = random_state

        # apple M chip device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        print(f"Using device: {self.device}")

        # tokenizer
        self.tokenizer = None

        # model
        self.model = None

        # label encoder
        self.label_encoder = None
        self.classes_ = None
        self.num_classes = None

        # training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def _initialize_model(self):
        # load pre-trained tokenizer
        print(f"\nLoading tokenizer: {self.model_name}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        # load pre-trained model with classification head
        print(f"Loading model: {self.model_name}")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )

        # move to device
        self.model.to(self.device)

    def _tokenize_texts(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokenize texts and create attention masks

        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids']
        attention_masks = encodings['attention_mask']

        return input_ids, attention_masks

    def _create_dataloader(
            self,
            input_ids: torch.Tensor,
            attention_masks: torch.Tensor,
            labels: torch.Tensor,
            batch_size: int,
            shuffle: bool = True
    ) -> DataLoader:
        # create dataloader for batching

        dataset = TensorDataset(input_ids, attention_masks, labels)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

        return dataloader

    def _train_epoch(
            self,
            dataloader: DataLoader,
            optimizer,
            scheduler
    ) -> Tuple[float, float]:
        # train for one epoch

        self.model.train()

        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(dataloader, desc="Training", leave=False)

        for batch in progress_bar:
            # unpack batch
            b_input_ids = batch[0].to(self.device)
            b_attention_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            # zero gradients
            optimizer.zero_grad()

            # forward pass
            outputs = self.model(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask,
                labels=b_labels
            )

            loss = outputs.loss
            logits = outputs.logits

            # backward pass
            loss.backward()

            # clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # update weights
            optimizer.step()
            scheduler.step()

            # track metrics
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == b_labels).sum().item()
            total_predictions += b_labels.size(0)

            # update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct_predictions / total_predictions
            })

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy

    def _evaluate_epoch(
            self,
            dataloader: DataLoader
    ) -> Tuple[float, float]:
        # evaluate on validation set

        self.model.eval()

        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in dataloader:
                # unpack batch
                b_input_ids = batch[0].to(self.device)
                b_attention_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # forward pass
                outputs = self.model(
                    input_ids=b_input_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels
                )

                loss = outputs.loss
                logits = outputs.logits

                # track metrics
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct_predictions += (preds == b_labels).sum().item()
                total_predictions += b_labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy

    def fit(
            self,
            texts: List[str],
            labels: np.ndarray,
            validation_data: Optional[Tuple[List[str], np.ndarray]] = None,
            batch_size: int = 16,
            epochs: int = 4,
            early_stopping_patience: int = 2,
            verbose: int = 1
    ) -> 'TransformerClassifier':

        print(f"\nTraining Transformer classifier...")
        print(f"Model: {self.model_name}")
        print(f"Max length: {self.max_length}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")

        # encode labels
        self.classes_ = np.unique(labels)
        self.num_classes = len(self.classes_)
        self.label_encoder = {label: idx for idx, label in enumerate(self.classes_)}

        print(f"\nNumber of classes: {self.num_classes}")
        print(f"Classes: {list(self.classes_)}")

        # initialize model
        self._initialize_model()

        # tokenize training data
        print("\nTokenizing training data...")
        train_input_ids, train_attention_masks = self._tokenize_texts(texts)
        train_labels_encoded = torch.tensor(
            [self.label_encoder[label] for label in labels]
        )

        print(f"Training data shape: {train_input_ids.shape}")

        # create training dataloader
        train_dataloader = self._create_dataloader(
            train_input_ids,
            train_attention_masks,
            train_labels_encoded,
            batch_size=batch_size,
            shuffle=True
        )

        # prepare validation data if provided
        val_dataloader = None
        if validation_data is not None:
            val_texts, val_labels = validation_data

            print("Tokenizing validation data...")
            val_input_ids, val_attention_masks = self._tokenize_texts(val_texts)
            val_labels_encoded = torch.tensor(
                [self.label_encoder[label] for label in val_labels]
            )

            print(f"Validation data shape: {val_input_ids.shape}")

            val_dataloader = self._create_dataloader(
                val_input_ids,
                val_attention_masks,
                val_labels_encoded,
                batch_size=batch_size,
                shuffle=False
            )

        # setup optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=1e-8
        )

        # setup learning rate scheduler
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        # training loop
        print(f"\nTraining for {epochs} epochs...")
        print("=" * 60)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)

            # train
            train_loss, train_acc = self._train_epoch(
                train_dataloader,
                optimizer,
                scheduler
            )

            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

            # validate
            if val_dataloader is not None:
                val_loss, val_acc = self._evaluate_epoch(val_dataloader)

                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)

                print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

                # early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print("✓ New best validation loss!")
                else:
                    patience_counter += 1
                    print(f"Patience: {patience_counter}/{early_stopping_patience}")

                    if patience_counter >= early_stopping_patience:
                        print(f"\nEarly stopping triggered after epoch {epoch + 1}")
                        break

        print("\n" + "=" * 60)
        print("✅ Training complete!")

        return self

    def predict(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        # tokenize
        input_ids, attention_masks = self._tokenize_texts(texts)

        # create dataloader
        dummy_labels = torch.zeros(len(texts), dtype=torch.long)
        dataloader = self._create_dataloader(
            input_ids,
            attention_masks,
            dummy_labels,
            batch_size=batch_size,
            shuffle=False
        )

        # predict
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch in dataloader:
                b_input_ids = batch[0].to(self.device)
                b_attention_mask = batch[1].to(self.device)

                outputs = self.model(
                    input_ids=b_input_ids,
                    attention_mask=b_attention_mask
                )

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_predictions.extend(preds)

        # decode labels
        label_decoder = {idx: label for label, idx in self.label_encoder.items()}
        predictions = np.array([label_decoder[idx] for idx in all_predictions])

        return predictions

    def predict_proba(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        # tokenize
        input_ids, attention_masks = self._tokenize_texts(texts)

        # create dataloader
        dummy_labels = torch.zeros(len(texts), dtype=torch.long)
        dataloader = self._create_dataloader(
            input_ids,
            attention_masks,
            dummy_labels,
            batch_size=batch_size,
            shuffle=False
        )

        # predict
        self.model.eval()
        all_probabilities = []

        with torch.no_grad():
            for batch in dataloader:
                b_input_ids = batch[0].to(self.device)
                b_attention_mask = batch[1].to(self.device)

                outputs = self.model(
                    input_ids=b_input_ids,
                    attention_mask=b_attention_mask
                )

                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probabilities.extend(probs)

        return np.array(all_probabilities)

    def get_training_history(self) -> dict:
        # return training history
        if not self.history['train_loss']:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self.history

    def save(self, filepath: str) -> None:
        # create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # save model
        model_dir = filepath.replace('.pkl', '_model')
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

        # save metadata
        metadata = {
            'label_encoder': self.label_encoder,
            'classes': self.classes_,
            'num_classes': self.num_classes,
            'model_name': self.model_name,
            'max_length': self.max_length,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'history': self.history
        }

        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"✅ Saved model to {model_dir}")
        print(f"✅ Saved metadata to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'TransformerClassifier':
        # load metadata
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)

        # create instance
        instance = cls(
            model_name=metadata['model_name'],
            max_length=metadata['max_length'],
            learning_rate=metadata['learning_rate'],
            warmup_steps=metadata['warmup_steps']
        )

        # restore metadata
        instance.label_encoder = metadata['label_encoder']
        instance.classes_ = metadata['classes']
        instance.num_classes = metadata['num_classes']
        instance.history = metadata.get('history', {})

        # load model and tokenizer
        model_dir = filepath.replace('.pkl', '_model')
        instance.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        instance.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        instance.model.to(instance.device)

        print(f"✅ Loaded model from {model_dir}")
        print(f"✅ Loaded metadata from {filepath}")

        return instance


if __name__ == "__main__":
    # test the Transformer classifier
    print("Transformer Classifier - Test Run")
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
                  ] * 50

    train_labels = np.array(
        ['Technology', 'Business', 'Sports'] * 2 * 50
    )

    test_texts = [
        "deep learning neural network training",
        "financial market investment portfolio",
        "basketball game championship final"
    ]

    print(f"\nTraining data: {len(train_texts)} samples")
    print(f"Test data: {len(test_texts)} samples")
    print(f"Classes: {np.unique(train_labels)}")

    # train Transformer model
    model = TransformerClassifier(
        model_name='distilbert-base-uncased',
        max_length=128,
        learning_rate=2e-5
    )

    model.fit(
        train_texts,
        train_labels,
        batch_size=8,
        epochs=2,
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

    model_path = project_root / 'models' / 'test_transformer_model.pkl'
    model.save(str(model_path))

    # load and test
    loaded_model = TransformerClassifier.load(str(model_path))
    loaded_predictions = loaded_model.predict(test_texts)

    print(f"\nOriginal predictions: {predictions}")
    print(f"Loaded predictions:   {loaded_predictions}")
    print(f"Match: {np.array_equal(predictions, loaded_predictions)}")

    print("\n" + "=" * 60)
    print("✅ Transformer classifier test complete!")
    print("=" * 60)
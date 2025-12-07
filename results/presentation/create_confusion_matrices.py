"""
Generate confusion matrices for all 4 models with custom formatting.
Format: 5:4 aspect ratio, ColorBrewer colors
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json
from pathlib import Path

# Import model classes
from src.ngrams_model import NGramClassifier
from src.embeddings_model import EmbeddingsClassifier
from src.lstm_model import LSTMClassifier
from src.transformer_model import TransformerClassifier
from src.data_splitting import load_splits
from src.preprocessing import TextPreprocessor

# ColorBrewer YlOrRd color scheme (sequential)
# https://colorbrewer2.org/#type=sequential&scheme=YlOrRd&n=9
COLORBREWER_CMAP = 'YlOrRd'

def create_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """
    Create and save a confusion matrix with specified formatting.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Title for the plot
        save_path: Path to save the figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    # Create figure with 5:4 aspect ratio
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot confusion matrix with ColorBrewer colors
    im = ax.imshow(cm, interpolation='nearest', cmap=COLORBREWER_CMAP)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')

    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    # Adjust layout to prevent label cutoff
    fig.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved confusion matrix to {save_path}")

def main():
    """Generate confusion matrices for all models."""

    # Load validation data
    print("Loading validation data...")
    train_df, val_df, test_df = load_splits('../../data/processed')

    # Get class names (sorted for consistency)
    class_names = sorted(val_df['unified_topic'].unique())
    val_true_labels = val_df['unified_topic'].values

    # Create output directory
    output_dir = Path('')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. N-Grams (Logistic Regression)
    print("\n1. Processing N-Grams model...")
    ngrams_model = NGramClassifier.load('../../models/ngrams_logistic.pkl')
    ngrams_pred = ngrams_model.predict(val_df['content'])
    create_confusion_matrix(
        val_true_labels,
        ngrams_pred,
        class_names,
        'N-Grams',
        output_dir / 'ngrams_confusion_matrix.png'
    )

    # 2. Word2Vec Embeddings (Logistic Regression)
    print("\n2. Processing Embeddings Word2Vec model...")
    preprocessor = TextPreprocessor(remove_stopwords=True, min_token_length=2)
    val_texts_preprocessed = preprocessor.transform(val_df['content'].tolist())

    embeddings_model = EmbeddingsClassifier.load('../../models/embeddings_word2vec_logistic.pkl')
    embeddings_pred = embeddings_model.predict(val_texts_preprocessed)
    create_confusion_matrix(
        val_true_labels,
        embeddings_pred,
        class_names,
        'Embeddings Word2Vec',
        output_dir / 'embeddings_confusion_matrix.png'
    )

    # 3. LSTM
    print("\n3. Processing LSTM model...")
    preprocessor_lstm = TextPreprocessor(remove_stopwords=False, min_token_length=2)
    val_texts_lstm = preprocessor_lstm.transform(val_df['content'].tolist())

    lstm_model = LSTMClassifier.load('../../models/lstm_model.pkl')
    lstm_pred = lstm_model.predict(val_texts_lstm)
    create_confusion_matrix(
        val_true_labels,
        lstm_pred,
        class_names,
        'LSTM',
        output_dir / 'lstm_confusion_matrix.png'
    )

    # 4. Transformer
    print("\n4. Processing Transformer model...")
    # Transformer uses raw text (no preprocessing)
    val_texts_raw = val_df['content'].tolist()

    transformer_model = TransformerClassifier.load('../../models/transformer.pkl')
    transformer_pred = transformer_model.predict(val_texts_raw, batch_size=16)
    create_confusion_matrix(
        val_true_labels,
        transformer_pred,
        class_names,
        'Transformer',
        output_dir / 'transformer_confusion_matrix.png'
    )

    print("\n✓ All confusion matrices generated successfully!")
    print(f"✓ Saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()

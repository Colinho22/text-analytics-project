"""Generate confusion matrices for presentation"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path

# Import models and utilities
from src.ngrams_model import NGramClassifier
from src.embeddings_model import EmbeddingsClassifier
from src.lstm_model import LSTMClassifier
from src.transformer_model import TransformerClassifier
from src.data_splitting import load_splits
from src.preprocessing import TextPreprocessor

# Load validation data
print("Loading data...")
train_df, val_df, test_df = load_splits('../../data/processed')
class_names = sorted(val_df['unified_topic'].unique())
val_true = val_df['unified_topic'].values

# Create output directory
output_dir = Path('')
output_dir.mkdir(parents=True, exist_ok=True)

def plot_cm(cm, title, filename):
    """Plot confusion matrix with 5:4 aspect ratio and ColorBrewer colors"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use ColorBrewer YlOrRd colormap
    im = ax.imshow(cm, interpolation='nearest', cmap='YlOrRd')
    ax.figure.colorbar(im, ax=ax)

    # Labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# 1. N-Grams
print("\n1. N-Grams...")
model = NGramClassifier.load('../../models/ngrams_logistic.pkl')
pred = model.predict(val_df['content'])
cm = confusion_matrix(val_true, pred, labels=class_names)
plot_cm(cm, 'N-Grams', 'ngrams_confusion_matrix.png')

# 2. Embeddings
print("\n2. Embeddings Word2Vec...")
preprocessor = TextPreprocessor(remove_stopwords=True, min_token_length=2)
val_texts = preprocessor.transform(val_df['content'].tolist())
model = EmbeddingsClassifier.load('../../models/embeddings_word2vec_logistic.pkl')
pred = model.predict(val_texts)
cm = confusion_matrix(val_true, pred, labels=class_names)
plot_cm(cm, 'Embeddings Word2Vec', 'embeddings_confusion_matrix.png')

# 3. LSTM
print("\n3. LSTM...")
preprocessor = TextPreprocessor(remove_stopwords=False, min_token_length=2)
val_texts = preprocessor.transform(val_df['content'].tolist())
model = LSTMClassifier.load('../../models/lstm_model.pkl')
pred = model.predict(val_texts)
cm = confusion_matrix(val_true, pred, labels=class_names)
plot_cm(cm, 'LSTM', 'lstm_confusion_matrix.png')

# 4. Transformer
print("\n4. Transformer...")
model = TransformerClassifier.load('../../models/transformer.pkl')
pred = model.predict(val_df['content'].tolist(), batch_size=16)
cm = confusion_matrix(val_true, pred, labels=class_names)
plot_cm(cm, 'Transformer', 'transformer_confusion_matrix.png')

print("\nDone! All confusion matrices saved to results/presentation/")
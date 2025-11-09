# Training script for LSTM model.
# 1. Loads train/val/test splits
# 2. Preprocesses the text
# 3. Trains LSTM with embeddings from scratch
# 4. Evaluates on validation set
# 5. Creates training history plots
# 6. Saves model and results

# import libraries
import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# add src to path for imports
current_file = Path(__file__).resolve()
if current_file.parent.name == 'src':
    project_root = current_file.parent.parent
    src_path = current_file.parent
else:
    project_root = current_file.parent
    src_path = project_root / 'src'

sys.path.insert(0, str(src_path))

from preprocessing import TextPreprocessor
from data_splitting import load_splits
from evaluation import evaluate_model
from lstm_model import LSTMClassifier

# set plotting style
plt.style.use('default')
sns.set_palette("Set2")


def plot_training_history(history_dict: dict, output_path: str):
    # plot training and validation metrics over epochs

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history_dict['loss']) + 1)

    # loss plot
    axes[0].plot(epochs, history_dict['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # accuracy plot
    axes[1].plot(epochs, history_dict['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history_dict['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved training history plot to {output_path}")


# training pipeline
def main():
    print("\n" + "=" * 60)
    print("LSTM MODEL TRAINING")
    print("=" * 60)

    # config
    config = {
        'max_features': 20000,
        'max_sequence_length': 250,
        'embedding_dim': 200,
        'lstm_units': 128,
        'num_layers': 2,
        'dropout': 0.4,
        'recurrent_dropout': 0.2,
        'batch_size': 32,
        'epochs': 20,
        'early_stopping_patience': 3,
        'validation_split': 0.1,
        'remove_stopwords': False,
        'min_token_length': 2
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # paths
    data_dir = project_root / 'data' / 'processed'
    models_dir = project_root / 'models'
    results_dir = project_root / 'results' / 'lstm'

    # create dirs
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # step 1: Load data
    print("\n" + "-" * 60)
    print("STEP 1: Loading data splits")
    print("-" * 60)

    train_df, val_df, test_df = load_splits(str(data_dir))

    print(f"\nDataset sizes:")
    print(f"  Train:      {len(train_df):,} samples")
    print(f"  Validation: {len(val_df):,} samples")
    print(f"  Test:       {len(test_df):,} samples")

    # get class names
    class_names = sorted(train_df['unified_topic'].unique())
    print(f"\nClasses ({len(class_names)}): {', '.join(class_names)}")

    # step 2: preprocess text
    print("\n" + "-" * 60)
    print("STEP 2: Preprocessing text")
    print("-" * 60)

    preprocessor = TextPreprocessor(
        remove_stopwords=config['remove_stopwords'],
        min_token_length=config['min_token_length']
    )

    print("\nPreprocessing training data...")
    train_texts = preprocessor.transform(train_df['content'].tolist())
    train_labels = train_df['unified_topic'].values

    print("Preprocessing validation data...")
    val_texts = preprocessor.transform(val_df['content'].tolist())
    val_labels = val_df['unified_topic'].values

    print("✅ Preprocessing complete!")

    # preview
    print("\nPreprocessing preview:")
    print(f"Original:  {train_df['content'].iloc[0][:100]}...")
    print(f"Processed: {train_texts[0][:100]}...")

    # step 3: train model
    print("\n" + "-" * 60)
    print("STEP 3: Training LSTM Model")
    print("-" * 60)

    start_time = time.time()

    model = LSTMClassifier(
        vocab_size=config['max_features'],
        max_sequence_length=config['max_sequence_length'],
        embedding_dim=config['embedding_dim'],
        lstm_units=config['lstm_units'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        recurrent_dropout=config['recurrent_dropout']
    )

    model.fit(
        train_texts,
        train_labels,
        validation_data=(val_texts, val_labels),
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        early_stopping_patience=config['early_stopping_patience'],
        verbose=1
    )

    train_time = time.time() - start_time
    print(f"\n⏱️  Training time: {train_time:.2f} seconds ({train_time / 60:.1f} minutes)")

    # step 4: plot training history
    print("\n" + "-" * 60)
    print("STEP 4: Plotting training history")
    print("-" * 60)

    history_dict = model.get_training_history()
    history_plot_path = results_dir / 'lstm_training_history.png'
    plot_training_history(history_dict, str(history_plot_path))

    # step 5: evaluate model
    print("\n" + "-" * 60)
    print("STEP 5: Evaluating model")
    print("-" * 60)

    predictions = model.predict(val_texts)

    metrics = evaluate_model(
        val_labels,
        predictions,
        class_names,
        method_name='lstm',
        output_dir=str(results_dir),
        save_plots=True,
        training_time_seconds=train_time
    )

    # save model
    model_path = models_dir / 'lstm_model.pkl'
    model.save(str(model_path))

    # step 6: compare with previous methods
    print("\n" + "-" * 60)
    print("STEP 6: Comparing with previous methods")
    print("-" * 60)

    # load previous results
    ngrams_metrics_file = project_root / 'results' / 'ngrams' / 'ngrams_logistic_metrics.json'
    embeddings_metrics_file = project_root / 'results' / 'embeddings' / 'embeddings_word2vec_logistic_metrics.json'

    comparison_data = {'LSTM': metrics}

    if ngrams_metrics_file.exists():
        import json
        with open(ngrams_metrics_file, 'r') as f:
            comparison_data['N-grams'] = json.load(f)

    if embeddings_metrics_file.exists():
        import json
        with open(embeddings_metrics_file, 'r') as f:
            comparison_data['Word2Vec'] = json.load(f)

    if len(comparison_data) > 1:
        comparison = pd.DataFrame({
            method: {
                'Accuracy': data['accuracy'],
                'Macro F1': data['macro_f1'],
                'Training Time (s)': data.get('training_time_seconds', 0)
            }
            for method, data in comparison_data.items()
        })

        print("\n" + "=" * 60)
        print("METHOD COMPARISON")
        print("=" * 60)
        print(comparison.to_string())

        # determine best method
        best_method = comparison.loc['Macro F1'].idxmax()
        best_f1 = comparison.loc['Macro F1'].max()
        print(f"\n🏆 Best method: {best_method}")
        print(f"   Macro F1: {best_f1:.4f}")

    # step 7: analyze per-class performance
    print("\n" + "-" * 60)
    print("STEP 7: Per-class performance")
    print("-" * 60)

    per_class_df = pd.DataFrame(metrics['per_class_metrics']).T
    per_class_df = per_class_df.sort_values('f1', ascending=False)

    print("\nF1 Scores by Category:")
    for topic, row in per_class_df.iterrows():
        print(f"  {topic:15s}: {row['f1']:.3f}")

    # final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    print("\n📊 Results:")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Macro F1:   {metrics['macro_f1']:.4f}")
    print(f"  Train Time: {train_time:.2f} seconds")

    print("\n📁 Generated files:")
    print(f"  Model:")
    print(f"    • {model_path}")
    print(f"    • {str(model_path).replace('.pkl', '_model.keras')}")
    print(f"\n  Results (in results/lstm/):")
    print(f"    • lstm_metrics.json")
    print(f"    • lstm_confusion_matrix.png")
    print(f"    • lstm_classification_report.txt")
    print(f"    • lstm_training_history.png")

    print("\n🎉 LSTM training complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
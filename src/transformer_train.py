# Training script for Transformer model.
# 1. Loads train/val/test splits
# 2. Trains DistilBERT with fine-tuning
# 3. Evaluates on validation set
# 4. Creates training history plots
# 5. Saves model and results

# import libraries
import pandas as pd
from pathlib import Path
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# suppress warnings
warnings.filterwarnings('ignore')

# auto-detect project root
current_file = Path(__file__).resolve()
if current_file.parent.name == 'src':
    project_root = current_file.parent.parent
    src_path = current_file.parent
else:
    project_root = current_file.parent
    src_path = project_root / 'src'

sys.path.insert(0, str(src_path))

from data_splitting import load_splits
from evaluation import evaluate_model
from transformer_model import TransformerClassifier

# set plotting style
plt.style.use('default')
sns.set_palette("Set2")


def plot_training_history(history_dict: dict, output_path: str):
    # plot training and validation metrics over epochs

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history_dict['train_loss']) + 1)

    # loss plot
    axes[0].plot(epochs, history_dict['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if history_dict['val_loss']:
        axes[0].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # accuracy plot
    axes[1].plot(epochs, history_dict['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    if history_dict['val_accuracy']:
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


def main():
    print("\n" + "=" * 60)
    print("TRANSFORMER MODEL TRAINING (DistilBERT)")
    print("=" * 60)

    # config
    # Note: batch_size can be increased to 32 or 64 on M3 Pro/Max for faster training
    config = {
        'model_name': 'distilbert-base-uncased',
        'max_length': 512,
        'learning_rate': 2e-5,
        'warmup_steps': 500,
        'batch_size': 16,
        'epochs': 4,
        'early_stopping_patience': 2
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # paths
    data_dir = project_root / 'data' / 'processed'
    models_dir = project_root / 'models'
    results_dir = project_root / 'results' / 'transformer'

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

    # step 2: prepare data
    print("\n" + "-" * 60)
    print("STEP 2: Preparing data")
    print("-" * 60)

    print("\nExtracting texts and labels...")
    train_texts = train_df['content'].tolist()
    train_labels = train_df['unified_topic'].values

    val_texts = val_df['content'].tolist()
    val_labels = val_df['unified_topic'].values

    print("✅ Data preparation complete!")

    # preview
    print("\nData preview:")
    print(f"Sample text: {train_texts[0][:200]}...")
    print(f"Sample label: {train_labels[0]}")

    # step 3: train model
    print("\n" + "-" * 60)
    print("STEP 3: Training Transformer Model")
    print("-" * 60)

    start_time = time.time()

    model = TransformerClassifier(
        model_name=config['model_name'],
        max_length=config['max_length'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps']
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
    history_plot_path = results_dir / 'transformer_training_history.png'
    plot_training_history(history_dict, str(history_plot_path))

    # step 5: evaluate model
    print("\n" + "-" * 60)
    print("STEP 5: Evaluating model")
    print("-" * 60)

    print("Generating predictions on validation set...")
    predictions = model.predict(val_texts, batch_size=config['batch_size'])

    metrics = evaluate_model(
        val_labels,
        predictions,
        class_names,
        method_name='transformer',
        output_dir=str(results_dir),
        save_plots=True,
        training_time_seconds=train_time
    )

    # step 6: save model
    print("\n" + "-" * 60)
    print("STEP 6: Saving model")
    print("-" * 60)

    model_path = models_dir / 'transformer.pkl'
    model.save(str(model_path))

    # step 7: compare with previous methods
    print("\n" + "-" * 60)
    print("STEP 7: Comparing with previous methods")
    print("-" * 60)

    # load previous results
    ngrams_metrics_file = project_root / 'results' / 'ngrams' / 'ngrams_logistic_metrics.json'
    embeddings_metrics_file = project_root / 'results' / 'embeddings' / 'embeddings_word2vec_logistic_metrics.json'
    lstm_metrics_file = project_root / 'results' / 'lstm' / 'lstm_metrics.json'

    comparison_data = {'Transformer': metrics}

    if ngrams_metrics_file.exists():
        import json
        with open(ngrams_metrics_file, 'r') as f:
            comparison_data['N-grams'] = json.load(f)

    if embeddings_metrics_file.exists():
        import json
        with open(embeddings_metrics_file, 'r') as f:
            comparison_data['Word2Vec'] = json.load(f)

    if lstm_metrics_file.exists():
        import json
        with open(lstm_metrics_file, 'r') as f:
            comparison_data['LSTM'] = json.load(f)

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

        # create comparison bar plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # accuracy comparison
        acc_data = comparison.loc['Accuracy'].sort_values(ascending=False)
        axes[0].bar(range(len(acc_data)), acc_data.values, color='steelblue', alpha=0.7)
        axes[0].set_xticks(range(len(acc_data)))
        axes[0].set_xticklabels(acc_data.index, rotation=45, ha='right')
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([0, 1])

        # add value labels on bars
        for i, v in enumerate(acc_data.values):
            axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)

        # F1 comparison
        f1_data = comparison.loc['Macro F1'].sort_values(ascending=False)
        axes[1].bar(range(len(f1_data)), f1_data.values, color='coral', alpha=0.7)
        axes[1].set_xticks(range(len(f1_data)))
        axes[1].set_xticklabels(f1_data.index, rotation=45, ha='right')
        axes[1].set_ylabel('Macro F1', fontsize=12)
        axes[1].set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_ylim([0, 1])

        # add value labels on bars
        for i, v in enumerate(f1_data.values):
            axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        comparison_plot_path = results_dir / 'model_comparison.png'
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✅ Saved comparison plot to {comparison_plot_path}")

    # step 8: analyze per-class performance
    print("\n" + "-" * 60)
    print("STEP 8: Per-class performance")
    print("-" * 60)

    per_class_df = pd.DataFrame(metrics['per_class_metrics']).T
    per_class_df = per_class_df.sort_values('f1', ascending=False)

    print("\nF1 Scores by Category:")
    for topic, row in per_class_df.iterrows():
        print(f"  {topic:15s}: {row['f1']:.3f}")

    # find best and worst performing classes
    best_class = per_class_df.index[0]
    worst_class = per_class_df.index[-1]
    print(f"\n🏆 Best performing: {best_class} (F1: {per_class_df.loc[best_class, 'f1']:.3f})")
    print(f"⚠️  Worst performing: {worst_class} (F1: {per_class_df.loc[worst_class, 'f1']:.3f})")

    # final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    print("\n📊 Results:")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Macro F1:   {metrics['macro_f1']:.4f}")
    print(f"  Train Time: {train_time:.2f} seconds ({train_time / 60:.1f} minutes)")

    print("\n📁 Generated files:")
    print(f"  Model:")
    print(f"    • {model_path}")
    print(f"    • {str(model_path).replace('.pkl', '_model/')}")
    print(f"\n  Results (in results/transformer/):")
    print(f"    • transformer_metrics.json")
    print(f"    • transformer_confusion_matrix.png")
    print(f"    • transformer_classification_report.txt")
    print(f"    • transformer_training_history.png")
    print(f"    • model_comparison.png")

    print("\n🎉 Transformer training complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
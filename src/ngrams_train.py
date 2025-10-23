# Training script for N-Grams baseline models.
# 1. Loads train/val/test splits
# 2. Preprocesses the text
# 3. Trains both Naive Bayes and Logistic Regression models
# 4. Evaluates on validation set
# 5. Saves models and results

# import libraries
import pandas as pd
from pathlib import Path
import time
import sys

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
from ngrams_model import NGramClassifier


# training pipeline
def main():

    print("\n" + "=" * 60)
    print("N-GRAMS BASELINE TRAINING")
    print("=" * 60)

    # config
    config = {
        'ngram_range': (1, 3),
        'max_features': 10000,
        'min_df': 2,
        'max_df': 0.95,
        'remove_stopwords': False,
        'min_token_length': 2
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # paths
    data_dir = project_root / 'data' / 'processed'
    models_dir = project_root / 'models'
    results_dir = project_root / 'results' / 'ngrams'

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

    # step 3: train models
    print("\n" + "-" * 60)
    print("STEP 3: Training models")
    print("-" * 60)

    results = {}

    # train naive bayes
    print("\n" + "=" * 60)
    print("TRAINING NAIVE BAYES")
    print("=" * 60)

    start_time = time.time()

    nb_model = NGramClassifier(
        classifier_type='naive_bayes',
        ngram_range=config['ngram_range'],
        max_features=config['max_features'],
        min_df=config['min_df'],
        max_df=config['max_df']
    )

    nb_model.fit(train_texts, train_labels)

    nb_train_time = time.time() - start_time
    print(f"\n⏱️  Training time: {nb_train_time:.2f} seconds")

    # evaluate naive bayes
    print("\n" + "-" * 60)
    print("EVALUATING NAIVE BAYES")
    print("-" * 60)

    nb_predictions = nb_model.predict(val_texts)

    nb_metrics = evaluate_model(
        val_labels,
        nb_predictions,
        class_names,
        method_name='ngrams_naive_bayes',
        output_dir=str(results_dir),
        save_plots=True,
        training_time_seconds=nb_train_time  # pass training time
    )

    results['naive_bayes'] = nb_metrics

    # save naive bayes model
    nb_model_path = models_dir / 'ngrams_naive_bayes.pkl'
    nb_model.save(str(nb_model_path))

    # train logistic regression
    print("\n" + "=" * 60)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 60)

    start_time = time.time()

    lr_model = NGramClassifier(
        classifier_type='logistic',
        ngram_range=config['ngram_range'],
        max_features=config['max_features'],
        min_df=config['min_df'],
        max_df=config['max_df']
    )

    lr_model.fit(train_texts, train_labels)

    lr_train_time = time.time() - start_time
    print(f"\n⏱️  Training time: {lr_train_time:.2f} seconds")

    # evaluate logistic regression
    print("\n" + "-" * 60)
    print("EVALUATING LOGISTIC REGRESSION")
    print("-" * 60)

    lr_predictions = lr_model.predict(val_texts)

    lr_metrics = evaluate_model(
        val_labels,
        lr_predictions,
        class_names,
        method_name='ngrams_logistic',
        output_dir=str(results_dir),
        save_plots=True,
        training_time_seconds=lr_train_time  # Pass training time
    )

    results['logistic'] = lr_metrics

    # save logistic regression model
    lr_model_path = models_dir / 'ngrams_logistic.pkl'
    lr_model.save(str(lr_model_path))

    # step 4: compare results
    print("\n" + "=" * 60)
    print("COMPARISON: NAIVE BAYES vs LOGISTIC REGRESSION")
    print("=" * 60)

    comparison = pd.DataFrame({
        'Naive Bayes': {
            'Accuracy': results['naive_bayes']['accuracy'],
            'Macro F1': results['naive_bayes']['macro_f1'],
            'Training Time (s)': results['naive_bayes']['training_time_seconds']
        },
        'Logistic Regression': {
            'Accuracy': results['logistic']['accuracy'],
            'Macro F1': results['logistic']['macro_f1'],
            'Training Time (s)': results['logistic']['training_time_seconds']
        }
    })

    print("\n" + comparison.to_string())

    # determine winner
    if results['logistic']['macro_f1'] > results['naive_bayes']['macro_f1']:
        winner = 'Logistic Regression'
        best_model = lr_model
    else:
        winner = 'Naive Bayes'
        best_model = nb_model

    print(f"\n🏆 Best model: {winner}")
    print(f"   F1 Score: {max(results['logistic']['macro_f1'], results['naive_bayes']['macro_f1']):.4f}")

    # step 5: analyze top features (logistic regression only)
    print("\n" + "=" * 60)
    print("TOP FEATURES PER CLASS (Logistic Regression)")
    print("=" * 60)

    for class_name in class_names:
        print(f"\n{class_name}:")
        top_features = lr_model.get_top_features(class_name, top_n=10)
        for i, (feature, weight) in enumerate(top_features, 1):
            print(f"  {i:2d}. {feature:20s}  (weight: {weight:6.3f})")

    # save top features to file
    features_file = results_dir / 'ngrams_top_features.txt'
    with open(features_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TOP FEATURES PER CLASS (Logistic Regression)\n")
        f.write("=" * 60 + "\n\n")

        for class_name in class_names:
            f.write(f"{class_name}:\n")
            top_features = lr_model.get_top_features(class_name, top_n=20)
            for i, (feature, weight) in enumerate(top_features, 1):
                f.write(f"  {i:2d}. {feature:20s}  (weight: {weight:6.3f})\n")
            f.write("\n")

    print(f"\n✅ Saved top features to {features_file}")

    # final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    print("\n📁 Generated files:")
    print(f"  Models:")
    print(f"    • {nb_model_path}")
    print(f"    • {lr_model_path}")
    print(f"\n  Results (in results/ngrams/):")
    print(f"    • ngrams_naive_bayes_metrics.json")
    print(f"    • ngrams_naive_bayes_confusion_matrix.png")
    print(f"    • ngrams_naive_bayes_classification_report.txt")
    print(f"    • ngrams_logistic_metrics.json")
    print(f"    • ngrams_logistic_confusion_matrix.png")
    print(f"    • ngrams_logistic_classification_report.txt")
    print(f"    • ngrams_top_features.txt")

    print("\n🎉 N-Grams baseline training complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
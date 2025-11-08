# Training script for Word2Vec Embeddings model.
# 1. Loads train/val/test splits
# 2. Preprocesses the text
# 3. Trains Word2Vec + Logistic Regression classifier
# 4. Evaluates on validation set
# 5. Creates T-SNE visualizations
# 6. Saves models and results

# import libraries
import numpy as np
from pathlib import Path
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

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
from embeddings_model import EmbeddingsClassifier

# set plotting style
plt.style.use('default')
sns.set_palette("Set2")


def create_tsne_word_embeddings(
        model: EmbeddingsClassifier,
        train_texts: list,
        train_labels: np.ndarray,
        class_names: list,
        output_path: str,
        n_words_per_class: int = 20
):
    # create T-SNE visualization of word embeddings colored by category

    print("\nCreating T-SNE visualization of word embeddings...")

    # get top words per category
    words_to_plot = []
    word_labels = []

    for class_name in class_names:
        top_words = model.get_category_words(
            train_texts,
            train_labels,
            class_name,
            top_n=n_words_per_class
        )
        for word, count in top_words:
            words_to_plot.append(word)
            word_labels.append(class_name)

    # get embeddings for these words
    embeddings = []
    valid_words = []
    valid_labels = []

    for word, label in zip(words_to_plot, word_labels):
        embedding = model.get_word_embedding(word)
        if embedding is not None:
            embeddings.append(embedding)
            valid_words.append(word)
            valid_labels.append(label)

    embeddings = np.array(embeddings)

    print(f"Plotting {len(embeddings)} words...")

    # apply T-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    # create plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # plot each category
    for class_name in class_names:
        mask = np.array(valid_labels) == class_name
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=class_name,
            alpha=0.6,
            s=100
        )

    # add word labels for a subset
    sample_indices = np.random.choice(
        len(valid_words),
        size=min(50, len(valid_words)),
        replace=False
    )

    for idx in sample_indices:
        ax.annotate(
            valid_words[idx],
            (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
            fontsize=8,
            alpha=0.7
        )

    ax.set_xlabel('T-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('T-SNE Dimension 2', fontsize=12)
    ax.set_title('T-SNE Visualization of Word Embeddings by Topic', fontsize=14, pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved word embeddings T-SNE to {output_path}")


def create_tsne_document_embeddings(
        model: EmbeddingsClassifier,
        texts: list,
        labels: np.ndarray,
        class_names: list,
        output_path: str,
        sample_size: int = 2000
):
    # create T-SNE visualization of document embeddings colored by true labels

    print("\nCreating T-SNE visualization of document embeddings...")

    # sample if too many documents
    if len(texts) > sample_size:
        print(f"Sampling {sample_size} documents for visualization...")
        indices = np.random.choice(len(texts), size=sample_size, replace=False)
        texts = [texts[i] for i in indices]
        labels = labels[indices]

    # tokenize texts
    tokenized_texts = model._tokenize_texts(texts)

    # create document embeddings
    embeddings = np.array([
        model._get_document_embedding(tokens)
        for tokens in tokenized_texts
    ])

    print(f"Applying T-SNE to {len(embeddings)} documents...")

    # apply T-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    # create plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # plot each category
    for class_name in class_names:
        mask = labels == class_name
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=class_name,
            alpha=0.5,
            s=20
        )

    ax.set_xlabel('T-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('T-SNE Dimension 2', fontsize=12)
    ax.set_title('T-SNE Visualization of Document Embeddings by Topic', fontsize=14, pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved document embeddings T-SNE to {output_path}")


def save_similar_words(
        model: EmbeddingsClassifier,
        class_names: list,
        sample_words: list,
        output_path: str
):
    # save most similar words for sample words and categories

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SIMILAR WORDS FROM WORD2VEC MODEL\n")
        f.write("=" * 60 + "\n\n")

        # sample words
        f.write("SAMPLE WORD SIMILARITIES\n")
        f.write("-" * 60 + "\n\n")

        for word in sample_words:
            f.write(f"{word}:\n")
            similar = model.find_similar_words(word, top_n=10)
            for i, (similar_word, score) in enumerate(similar, 1):
                f.write(f"  {i:2d}. {similar_word:20s}  (similarity: {score:.3f})\n")
            f.write("\n")

        # category representative words
        f.write("\n" + "=" * 60 + "\n")
        f.write("CATEGORY SIMILARITIES\n")
        f.write("=" * 60 + "\n\n")

        for class_name in class_names:
            f.write(f"\n{class_name}:\n")
            f.write("-" * 60 + "\n")

            # try a few representative words for this category
            category_words = {
                'Business': ['business', 'company', 'market'],
                'Entertainment': ['movie', 'music', 'entertainment'],
                'Health': ['health', 'medical', 'disease'],
                'Politics': ['politics', 'election', 'government'],
                'Science': ['science', 'research', 'study'],
                'Sports': ['sports', 'game', 'team'],
                'Technology': ['technology', 'software', 'computer']
            }

            for word in category_words.get(class_name, []):
                similar = model.find_similar_words(word, top_n=5)
                if similar:
                    f.write(f"\n  Similar to '{word}':\n")
                    for similar_word, score in similar:
                        f.write(f"    {similar_word:20s}  (similarity: {score:.3f})\n")

    print(f"✅ Saved similar words to {output_path}")


# training pipeline
def main():
    print("\n" + "=" * 60)
    print("WORD2VEC EMBEDDINGS TRAINING")
    print("=" * 60)

    # config
    config = {
        'vector_size': 200,
        'window': 5,
        'min_count': 5,
        'epochs': 30,
        'use_tfidf': False, #over-filtering
        'remove_stopwords': True,
        'min_token_length': 2
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # paths
    data_dir = project_root / 'data' / 'processed'
    models_dir = project_root / 'models'
    results_dir = project_root / 'results' / 'embeddings'

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
    print("STEP 3: Training Word2Vec + Logistic Regression")
    print("-" * 60)

    start_time = time.time()

    model = EmbeddingsClassifier(
        classifier_type='logistic',
        vector_size=config['vector_size'],
        window=config['window'],
        min_count=config['min_count'],
        epochs=config['epochs'],
        use_tfidf=config['use_tfidf']
    )

    model.fit(train_texts, train_labels)

    train_time = time.time() - start_time
    print(f"\n⏱️  Training time: {train_time:.2f} seconds")

    # step 4: evaluate model
    print("\n" + "-" * 60)
    print("STEP 4: Evaluating model")
    print("-" * 60)

    predictions = model.predict(val_texts)

    metrics = evaluate_model(
        val_labels,
        predictions,
        class_names,
        method_name='embeddings_word2vec_logistic',
        output_dir=str(results_dir),
        save_plots=True,
        training_time_seconds=train_time
    )

    # save model
    model_path = models_dir / 'embeddings_word2vec_logistic.pkl'
    model.save(str(model_path))

    # step 5: create visualizations
    print("\n" + "-" * 60)
    print("STEP 5: Creating visualizations")
    print("-" * 60)

    # T-SNE of word embeddings
    tsne_words_path = results_dir / 'embeddings_tsne_words.png'
    create_tsne_word_embeddings(
        model,
        train_texts,
        train_labels,
        class_names,
        str(tsne_words_path),
        n_words_per_class=20
    )

    # T-SNE of document embeddings
    tsne_docs_path = results_dir / 'embeddings_tsne_documents.png'
    create_tsne_document_embeddings(
        model,
        val_texts,
        val_labels,
        class_names,
        str(tsne_docs_path),
        sample_size=2000
    )

    # step 6: analyze similar words
    print("\n" + "-" * 60)
    print("STEP 6: Analyzing word similarities")
    print("-" * 60)

    # sample words to test
    sample_words = [
        'business', 'company', 'market',
        'movie', 'music', 'actor',
        'health', 'medical', 'patient',
        'election', 'president', 'vote',
        'research', 'scientist', 'study',
        'game', 'team', 'player',
        'technology', 'software', 'computer'
    ]

    print("\nSample word similarities:")
    for word in sample_words[:5]:
        similar = model.find_similar_words(word, top_n=5)
        if similar:
            print(f"\n{word}:")
            for similar_word, score in similar:
                print(f"  {similar_word:20s}: {score:.3f}")

    # save similar words to file
    similar_words_path = results_dir / 'embeddings_similar_words.txt'
    save_similar_words(
        model,
        class_names,
        sample_words,
        str(similar_words_path)
    )

    # step 7: top category words
    print("\n" + "-" * 60)
    print("STEP 7: Analyzing top category words")
    print("-" * 60)

    category_words_path = results_dir / 'embeddings_category_words.txt'

    with open(category_words_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TOP WORDS PER CATEGORY\n")
        f.write("=" * 60 + "\n\n")

        for class_name in class_names:
            print(f"\n{class_name}:")
            top_words = model.get_category_words(
                train_texts,
                train_labels,
                class_name,
                top_n=20
            )

            f.write(f"\n{class_name}:\n")
            f.write("-" * 60 + "\n")

            for i, (word, count) in enumerate(top_words, 1):
                output_line = f"  {i:2d}. {word:20s}  (count: {count:,})"
                print(output_line)
                f.write(output_line + "\n")

    print(f"\n✅ Saved category words to {category_words_path}")

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
    print(f"\n  Results (in results/embeddings/):")
    print(f"    • embeddings_word2vec_logistic_metrics.json")
    print(f"    • embeddings_word2vec_logistic_confusion_matrix.png")
    print(f"    • embeddings_word2vec_logistic_classification_report.txt")
    print(f"    • embeddings_tsne_words.png")
    print(f"    • embeddings_tsne_documents.png")
    print(f"    • embeddings_similar_words.txt")
    print(f"    • embeddings_category_words.txt")

    print("\n🎉 Word2Vec embeddings training complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
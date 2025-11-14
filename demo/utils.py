# import libraries
import pickle
import time
import json
import sys
from pathlib import Path
from datetime import datetime
import pdfplumber
import numpy as np

# add parent directory to path to import src modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

# import model classes
from lstm_model import LSTMClassifier
from transformer_model import TransformerClassifier
from preprocessing import TextPreprocessor


# load all 4 models
def load_models():
    models_dir = project_root / 'models'

    print("Loading models from:", models_dir)

    models = {}

    # load N-grams model (pickle - has vectorizer + classifier)
    print("Loading N-grams model...")
    with open(models_dir / "ngrams_logistic.pkl", "rb") as f:
        models["N-grams"] = pickle.load(f)

    # load Embeddings model (Word2Vec + classifier)
    print("Loading Embeddings model...")
    with open(models_dir / "embeddings_word2vec_logistic.pkl", "rb") as f:
        models["Embeddings"] = pickle.load(f)

    # load LSTM model (custom class)
    print("Loading LSTM model...")
    models["LSTM"] = LSTMClassifier.load(str(models_dir / "lstm_model.pkl"))

    # load Transformer model (custom class)
    print("Loading Transformer model...")
    models["Transformer"] = TransformerClassifier.load(str(models_dir / "transformer.pkl"))

    print("✅ All models loaded successfully!")
    return models


# extract text and returns cleaned string from PDF file
def extract_text_from_pdf(pdf_file):
    text = ""

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text.strip()


# run classification and return predictions & time
def classify_with_model(model, text, model_name):
    start_time = time.time()

    try:
        # different preprocessing for different models
        if model_name == "N-grams":
            # N-grams model is a dict with vectorizer and classifier
            vectorizer = model['vectorizer']
            classifier = model['classifier']

            # transform text and predict
            X = vectorizer.transform([text])
            prediction = classifier.predict(X)[0]

        elif model_name == "Embeddings":
            # embeddings model is a dict with word2vec, tfidf, and classifier
            word2vec_model = model['word2vec_model']
            classifier = model['classifier']
            use_tfidf = model.get('use_tfidf', False)

            # preprocess text
            preprocessor = TextPreprocessor(remove_stopwords=True, min_token_length=2)
            processed_text = preprocessor.transform([text])[0]
            tokens = processed_text.split()

            # get word vectors
            vectors = []
            for token in tokens:
                if token in word2vec_model.wv:
                    vectors.append(word2vec_model.wv[token])

            if len(vectors) > 0:
                # average the vectors
                doc_vector = np.mean(vectors, axis=0)

                # apply TF-IDF weighting if used during training
                if use_tfidf and 'idf_dict' in model:
                    idf_dict = model['idf_dict']
                    weights = [idf_dict.get(token, 1.0) for token in tokens if token in word2vec_model.wv]
                    if len(weights) > 0:
                        doc_vector = doc_vector * np.mean(weights)
            else:
                # no vectors found - use zero vector
                doc_vector = np.zeros(model['vector_size'])

            # predict
            prediction = classifier.predict([doc_vector])[0]

        elif model_name == "LSTM":
            # LSTM needs preprocessed text (tokenized)
            preprocessor = TextPreprocessor(remove_stopwords=False, min_token_length=2)
            processed_text = preprocessor.transform([text])
            prediction = model.predict(processed_text)[0]

        elif model_name == "Transformer":
            # Transformer uses raw text (no preprocessing)
            prediction = model.predict([text])[0]

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        end_time = time.time()
        time_ms = int((end_time - start_time) * 1000)

        return prediction, time_ms

    except Exception as e:
        print(f"❌ Error in {model_name}: {str(e)}")
        # return error indication
        end_time = time.time()
        time_ms = int((end_time - start_time) * 1000)
        return f"ERROR: {str(e)[:50]}", time_ms


# save classification to JSONL
def save_results(results_data):
    output_file = project_root / 'demo' / 'results' / 'classification_history.jsonl'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # add timestamp
    results_data["timestamp"] = datetime.now().isoformat()

    # append to JSONL file
    with open(output_file, "a") as f:
        f.write(json.dumps(results_data) + "\n")

    print(f"✅ Results saved to {output_file}")


# load results from JSONL file
def load_results_history():
    output_file = project_root / 'demo' / 'results' / 'classification_history.jsonl'

    if not output_file.exists():
        return []

    results = []
    with open(output_file, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    return results


# calculate summary statistics
def get_summary_stats(results_history):
    if not results_history:
        return None

    model_names = ["N-grams", "Embeddings", "LSTM", "Transformer"]
    stats = {
        "total_tests": len(results_history),
        "models": {}
    }

    # points system: 1st=4pts, 2nd=3pts, 3rd=2pts, 4th=1pt
    points_map = {1: 4, 2: 3, 3: 2, 4: 1}

    for model_name in model_names:
        wins = 0
        correct_count = 0
        total_time = 0
        total_rank = 0
        total_points = 0
        top2_count = 0

        for result in results_history:
            model_data = result["results"].get(model_name, {})
            rank = model_data.get("rank", 4)

            if rank == 1:
                wins += 1
            if rank <= 2:
                top2_count += 1
            if model_data.get("correct", False):
                correct_count += 1

            total_time += model_data.get("time_ms", 0)
            total_rank += rank
            total_points += points_map.get(rank, 0)

        num_tests = len(results_history)
        stats["models"][model_name] = {
            "wins": wins,
            "win_rate": (wins / num_tests) * 100 if num_tests else 0,
            "accuracy": (correct_count / num_tests) * 100 if num_tests else 0,
            "avg_time_ms": total_time / num_tests if num_tests else 0,
            "avg_rank": total_rank / num_tests if num_tests else 0,
            "total_points": total_points,
            "top2_rate": (top2_count / num_tests) * 100 if num_tests else 0
        }

    return stats
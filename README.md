# Text Analytics Project

This project uses four different machine learning approaches with the goal to classify texts into the seven topics:
**Business**, **Entertainment**, **Health**, **Politics**, **Science**, **Sports**, and **Technology**.

This is done by  [aggregating data](https://github.com/Colinho22/text-analytics-project/blob/main/src/data_download.py)
from **Hugging Face** datasets, **arXiv** papers, **CMU** book excerpts, and **Wikipedia** articles covering the topic
areas. Next, the [data is processed](https://github.com/Colinho22/text-analytics-project/blob/main/src/data_splitting.py),
the models are [trained](https://github.com/Colinho22/text-analytics-project/tree/main/models) and then
[analyzed](https://github.com/Colinho22/text-analytics-project/tree/main/notebooks). All culminates in the final
demo app, which allows for an interactive challenge between the models to see who is truly
the best.

**Model Overview**
1. **N-grams** - TF-IDF vectorization with Logistic Regression
2. **Embeddings** - Word2Vec embeddings with Logistic Regression
3. **LSTM** - Long Short-Term Memory recurrent neural network
4. **Transformer** - Fine-tuned DistilBERT model

---
## Table of Contents
  - [Setup & Usage](#setup--usage)
    - [Quick Start](#quick-start)
    - [Setup](#setup)
    - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Dataset](#dataset)
  - [Model Details](#model-details)
    - [Preprocessing](#preprocessing)
    - [Model Hyperparameters](#model-hyperparameters)
  - [Citation](#citation)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)


---
## Setup & Usage
This project was built and run using the following resources:
- built with Python 3.10.11
- pip package manager
- requirements.txt
- MacBook Air, 3M, 24GB, MacOS 26.1

This repository is available as is according to its [license](https://github.com/Colinho22/text-analytics-project/blob/main/LICENSE).
The compatibility with other setups is not guaranteed!

---
### Quick Start
Want to try the models without training?
1. Clone repo & install dependencies (Setup steps 1-3 below)
2. Download [transformer weights](https://drive.google.com/file/d/1Q8xCLgIWbKZlZFaVFjct7h0NuOXdl39r/view?usp=drive_link)
3. Run app.py
```bash
cd demo
streamlit run app.py
```

---
### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Colinho22/text-analytics-project.git
cd text-analytics-project
```

---
2. **Create virtual environment (recommended)**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

---
3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---
4. **Download NLTK data** (if needed)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---
5. **⚠️ IMPORTANT: Download Transformer Model Weights**

The transformer model weights (`model.safetensors`, 255MB) are not included in the repository due to file size. You have two options:

**Option A: Download Pre-trained Weights**
- Download from: [Google Drive](https://drive.google.com/file/d/1Q8xCLgIWbKZlZFaVFjct7h0NuOXdl39r/view?usp=drive_link)
- Extract to: `models/transformer_model/`
- Verify you have: `models/transformer_model/model.safetensors`

**Option B: Train from Scratch**

If you want to train the models yourself anyway, you can follow the usage flow below. Please note that the Transformer
training is optimized to use the Apple M-Series chip GPU and other chip architecture were not tested!

---
### Usage

After completing the setup section, you can continue with the usage flow. If you want to skip the training part, jump to
step **9**

6. **Download and Prepare Dataset**
```bash
python src/data_download.py
python src/data_splitting.py
```

---
7. **Train Individual Models**

Execute the training scripts for N-grams, Embeddings, LSTM, and Transformer ⚠️. Each training script will:
- Load the preprocessed data
- Train the model
- Evaluate on validation set
- Save the trained model to `models/`
- Generate visualizations in `results/`
- Print performance metrics

```bash
python src/ngrams_train.py
```

```bash
python src/embeddings_train.py
```

```bash
python src/lstm_train.py
```

```bash
python src/transformer_train.py
```
**Note ⚠️**: Transformer training is optimized to use Apple M-series chip GPU using PyTorch. Functionality for other chip
architectures not tested.

---
8. **Analyzing the Data and Models**

There are notebooks to analyze the dataset, processing and each model in more detail. They can be found 
[here](https://github.com/Colinho22/text-analytics-project/tree/main/notebooks).

---
9. **Running the Demo**

The interactive demo allows you to test all four models with your own PDF documents.

```bash
cd demo
streamlit run app.py
```

The app will be available at `http://localhost:8501`

**Demo Features:**
- Upload PDF documents
- Classify with all 4 models simultaneously
- See real-time predictions and inference time
- Rate model predictions for accuracy
- Save test results to JSONL file
- View aggregated statistics across tests

See [demo/DEMO_README.md](demo/DEMO_README.md) for detailed usage instructions.

---
## Project Structure

  - **`data/`** - Raw and processed datasets ([explore in notebooks](https://github.com/Colinho22/text-analytics-project/tree/main/notebooks))
  - **`src/`** - Training scripts and model implementations
  - **`models/`** - Trained model files (`.pkl`, `.keras`, transformer weights)
  - **`results/`** - Training metrics, confusion matrices, visualizations
  - **`notebooks/`** - Exploratory data analysis and model comparisons
  - **`demo/`** - Interactive Streamlit app ([see demo docs](https://github.com/Colinho22/text-analytics-project/blob/main/demo/DEMO_README.md))

---
## Model Details

### Preprocessing
- Lowercasing
- Punctuation removal
- Tokenization
- Stop word removal (optional, model-dependent)
- Minimum token length filtering

---
### Model Hyperparameters

**N-grams:**
- TF-IDF: ngram_range=(1, 3), max_features=10000
- Classifier: Logistic Regression, max_iter=1000

**Embeddings:**
- Word2Vec: vector_size=300, window=5, min_count=2
- Classifier: Logistic Regression

**LSTM:**
- Embedding dim: 128
- LSTM units: 128 (bidirectional)
- Dropout: 0.5
- Optimizer: Adam, lr=0.001

**Transformer:**
- Base model: distilbert-base-uncased
- Max length: 512 tokens
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 4 (with early stopping)

---

## Citation

If you use this project in your research or work, please cite:

```
@misc{text-analytics-project,
  author = {Colinho22},
  title = {Text Analytics Project},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Colinho22/text-analytics-project}
}
```

---
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
## Acknowledgments

- **Datasets**: arXiv, Hugging Face Datasets, Wikipedia
- **Libraries**: scikit-learn, gensim, PyTorch, Transformers, Streamlit
- **Models**: DistilBERT by Hugging Face

---
**Note**: This project was created as part of a Text Analytics course at the University of Applied Science of the
Grisons. The goal is to provide a comprehensive comparison of classical and modern NLP approaches for text classification.

# import libraries
import re
import string
from typing import List, Optional
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# cleaning function
def clean_text(text: str, remove_numbers: bool = False) -> str:
    if not isinstance(text, str):
        return ""

    # convert to lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # remove numbers if requested
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # remove punctuation but keep spaces
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    # remove extra whitespace
    text = ' '.join(text.split())

    return text


# stop words removal
def remove_stopwords(text: str, custom_stopwords: Optional[set] = None) -> str:

    # combine sklearn stopwords with any custom ones
    stopwords = set(ENGLISH_STOP_WORDS)
    if custom_stopwords:
        stopwords.update(custom_stopwords)

    # split, filter, rejoin
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]

    return ' '.join(filtered_words)

# short word removal
def filter_short_tokens(text: str, min_length: int = 2) -> str:
    words = text.split()
    filtered_words = [word for word in words if len(word) >= min_length]

    return ' '.join(filtered_words)

# preprocessing function
class TextPreprocessor:

    def __init__(
            self,
            lowercase: bool = True,
            remove_urls: bool = True,
            remove_numbers: bool = False,
            remove_stopwords: bool = False,
            min_token_length: int = 2,
            custom_stopwords: Optional[set] = None
    ):

        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_numbers = remove_numbers
        self.remove_stopwords_flag = remove_stopwords
        self.min_token_length = min_token_length
        self.custom_stopwords = custom_stopwords or set()

    def preprocess(self, text: str) -> str:

        # basic cleaning
        text = clean_text(text, remove_numbers=self.remove_numbers)

        # remove stopwords if requested
        if self.remove_stopwords_flag:
            text = remove_stopwords(text, custom_stopwords=self.custom_stopwords)

        # filter short tokens
        text = filter_short_tokens(text, min_length=self.min_token_length)

        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        return [self.preprocess(text) for text in texts]

    def fit(self, texts: List[str]):
        # no fitting needed for rule-based preprocessing
        return self

    def transform(self, texts: List[str]) -> List[str]:
        return self.preprocess_batch(texts)

    def fit_transform(self, texts: List[str]) -> List[str]:
        return self.fit(texts).transform(texts)


def get_default_preprocessor() -> TextPreprocessor:
    return TextPreprocessor(
        lowercase=True,
        remove_urls=True,
        remove_numbers=False,
        remove_stopwords=False,
        min_token_length=2,
        custom_stopwords=None
    )


if __name__ == "__main__":
    # test the preprocessing functions
    test_texts = [
        "Check out https://example.com for more info! Email me at test@email.com",
        "The quick brown fox jumps over the lazy dog.",
        "Machine Learning and AI are revolutionizing technology in 2024!",
        "   Extra    whitespace    should    be    removed   "
    ]

    print("Testing preprocessing functions:")
    print("=" * 60)

    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Original: {text}")
        print(f"Cleaned:  {clean_text(text)}")
        print(f"No stops: {remove_stopwords(clean_text(text))}")

    print("\n" + "=" * 60)
    print("\nTesting TextPreprocessor class:")
    print("=" * 60)

    # test with default settings
    preprocessor = get_default_preprocessor()
    processed = preprocessor.preprocess_batch(test_texts)

    for original, processed_text in zip(test_texts, processed):
        print(f"\nOriginal:  {original}")
        print(f"Processed: {processed_text}")

    print("\n" + "=" * 60)
    print("\nTesting with stopword removal:")
    print("=" * 60)

    # test with stopword removal
    preprocessor_no_stops = TextPreprocessor(remove_stopwords=True)
    processed_no_stops = preprocessor_no_stops.preprocess_batch(test_texts)

    for original, processed_text in zip(test_texts, processed_no_stops):
        print(f"\nOriginal:  {original}")
        print(f"Processed: {processed_text}")
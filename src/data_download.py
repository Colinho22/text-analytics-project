# import libs
import io
import json
import logging
import random
import shutil
import tarfile
import time
from pathlib import Path
import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm


# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# configuration constants (using absolute path)
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SAMPLES_PER_TOPIC_NEWS = 8000
SAMPLES_PER_TOPIC_ARXIV = 8000
SAMPLES_PER_TOPIC_BOOKS = 4000
SAMPLES_ENTERTAINMENT = 10000  # Rotten Tomatoes dataset size
SAMPLES_PER_TOPIC_WIKIPEDIA = 3000  # Wikipedia articles per topic

MIN_CONTENT_LENGTH = 50  # minimum characters
MAX_CONTENT_LENGTH = 10000  # maximum characters
ARXIV_RATE_LIMIT_DELAY = 3  # seconds between API calls
WIKIPEDIA_REQUEST_DELAY = 0.1  # seconds between Wikipedia API calls
RANDOM_SEED = 42


# unified topic mapping
UNIFIED_TOPICS = {
    'Technology': ['tech', 'ai', 'computing', 'software'],
    'Business': ['business', 'economics', 'finance', 'market'],
    'Science': ['physics', 'chemistry', 'biology', 'research'],
    'Health': ['medicine', 'health', 'medical', 'healthcare'],
    'Politics': ['politics', 'government', 'policy', 'law'],
    'Sports': ['sports', 'athletics', 'games', 'competition'],
    'Entertainment': ['movies', 'tv', 'film', 'entertainment', 'media']
}


# validate content meets min quality standards
def validate_content(text: str) -> bool:
    if pd.isna(text) or not isinstance(text, str):
        return False
    if len(text) < MIN_CONTENT_LENGTH or len(text) > MAX_CONTENT_LENGTH:
        return False
    return True


# remove duplicate content from df
def deduplicate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    initial_count = len(df)
    df = df.drop_duplicates(subset=['content'], keep='first')
    removed = initial_count - len(df)
    if removed > 0:
        logger.info(f"  Removed {removed} duplicate entries")
    return df


# sample from each topic in the df
def sample_by_topic(df: pd.DataFrame, samples_per_topic: int, topic_col: str = 'unified_topic') -> pd.DataFrame:
    sampled_dfs = []
    for topic in df[topic_col].unique():
        topic_df = df[df[topic_col] == topic]
        n_samples = min(samples_per_topic, len(topic_df))
        if n_samples > 0:
            topic_sample = topic_df.sample(n=n_samples, random_state=RANDOM_SEED)
            sampled_dfs.append(topic_sample)

    if sampled_dfs:
        return pd.concat(sampled_dfs, ignore_index=True)
    return pd.DataFrame()


# check packages
def check_dependencies():
    missing = []

    try:
        import arxiv
    except ImportError:
        missing.append('arxiv')

    try:
        from tqdm import tqdm
    except ImportError:
        missing.append('tqdm')

    if missing:
        logger.error(f"Missing required packages: {', '.join(missing)}")
        logger.error("Please install them using: pip install " + " ".join(missing))
        raise ImportError(f"Missing packages: {', '.join(missing)}")


# download ag news
def download_ag_news_targeted(samples_per_topic: int = SAMPLES_PER_TOPIC_NEWS) -> pd.DataFrame:
    logger.info("Downloading AG News...")

    dataset = load_dataset("ag_news")

    # combine train and test
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # map AG News categories to unified topics
    ag_to_unified = {
        3: 'Technology',
        2: 'Business',
        0: 'Politics',
        1: 'Sports'  # Now included!
    }

    full_df['unified_topic'] = full_df['label'].map(ag_to_unified)

    # rename and prepare content
    full_df = full_df.rename(columns={'text': 'content'})

    # validate content
    full_df['valid'] = full_df['content'].apply(validate_content)
    full_df = full_df[full_df['valid']]

    # sample equally from each topic
    result_df = sample_by_topic(full_df, samples_per_topic)

    # deduplicate
    result_df = deduplicate_dataframe(result_df)

    result_df['source'] = 'news'
    result_df = result_df[['content', 'unified_topic', 'source']]

    logger.info(f"✓ AG News: {len(result_df)} articles")
    logger.info(f"  Topics: {result_df['unified_topic'].value_counts().to_dict()}")

    return result_df


# download ArXiv papers and map topics
def download_arxiv_targeted(samples_per_topic: int = SAMPLES_PER_TOPIC_ARXIV) -> pd.DataFrame:
    logger.info("Downloading ArXiv papers...")

    try:
        import arxiv
    except ImportError:
        logger.error("arxiv package not installed. Please run: pip install arxiv")
        raise

    # map arxiv categories to unified topics (expanded for better coverage)
    arxiv_categories = {
        'Technology': ['cs.AI', 'cs.CL', 'cs.LG', 'cs.CV'],
        'Business': ['econ.GN', 'q-fin.EC'],
        'Science': ['physics.gen-ph', 'physics.app-ph', 'physics.bio-ph', 'cond-mat.soft', 'astro-ph.GA', 'chem.gen-chem'],
        'Health': ['q-bio.GN', 'q-bio.QM', 'q-bio.BM', 'q-bio.CB', 'q-bio.NC', 'q-bio.TO']
    }

    all_papers = []

    for topic, categories in arxiv_categories.items():
        logger.info(f"  Fetching {topic} papers...")
        topic_papers = []

        for category in tqdm(categories, desc=f"    {topic}", leave=False):
            try:
                # search for papers in this category
                search = arxiv.Search(
                    query=f"cat:{category}",
                    max_results=samples_per_topic // len(categories),
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )

                for result in search.results():
                    # use abstract as content, validate it
                    if result.summary and validate_content(result.summary):
                        topic_papers.append({
                            'content': result.summary,
                            'unified_topic': topic,
                            'source': 'scientific'
                        })

                # rate limit to avoid API blocks
                time.sleep(ARXIV_RATE_LIMIT_DELAY)

            except arxiv.ArxivError as e:
                logger.warning(f"    ArXiv API error for {category}: {e}")
                continue
            except Exception as e:
                logger.warning(f"    Could not fetch {category}: {e}")
                continue

        # sample to target amount if needed
        if len(topic_papers) > samples_per_topic:
            random.seed(RANDOM_SEED)
            topic_papers = random.sample(topic_papers, samples_per_topic)

        all_papers.extend(topic_papers)
        logger.info(f"    Collected {len(topic_papers)} papers")

    df = pd.DataFrame(all_papers)

    # deduplicate
    df = deduplicate_dataframe(df)

    logger.info(f"✓ ArXiv: {len(df)} papers")
    logger.info(f"  Topics: {df['unified_topic'].value_counts().to_dict()}")

    return df


# download book summaries and map topics
def download_books_targeted(samples_per_topic: int = SAMPLES_PER_TOPIC_BOOKS) -> pd.DataFrame:
    logger.info("Downloading Book Summaries...")

    # CMU Book Summaries URL
    url = "http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz"
    temp_dir = DATA_DIR / "books_temp"

    try:
        logger.info("  Downloading from CMU...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # extract tar file
        tar_file = tarfile.open(fileobj=io.BytesIO(response.content))
        tar_file.extractall(temp_dir)
        tar_file.close()

        # read the tab-separated file
        books_file = temp_dir / "booksummaries" / "booksummaries.txt"

        # columns: Wikipedia ID, Freebase ID, Book title, Author, Publication date, Genres, Summary
        df = pd.read_csv(
            books_file,
            sep='\t',
            names=['wiki_id', 'freebase_id', 'title', 'author', 'pub_date', 'genres', 'text'],
            on_bad_lines='skip'
        )

        # parse genres (JSON format)
        def extract_genres(genre_str):
            if pd.isna(genre_str):
                return []
            try:
                genres_dict = json.loads(genre_str)
                return list(genres_dict.values())
            except json.JSONDecodeError:
                return []

        df['genre_list'] = df['genres'].apply(extract_genres)

        # filter for books with summaries and genres
        df = df[df['text'].notna() & (df['genre_list'].str.len() > 0)].copy()

        # map genres to unified topics (expanded for better coverage)
        genre_to_topic = {
            # Technology
            'Science Fiction': 'Technology',
            'Speculative fiction': 'Technology',
            'Technology': 'Technology',
            'Cyberpunk': 'Technology',
            'Space opera': 'Technology',
            # Business
            'Business': 'Business',
            'Economics': 'Business',
            'Finance': 'Business',
            'Management': 'Business',
            'Marketing': 'Business',
            # Health
            'Medical fiction': 'Health',
            'Medicine': 'Health',
            'Health': 'Health',
            'Self-help': 'Health',
            'Psychology': 'Health',
            'Wellness': 'Health',
            # Science
            'Science': 'Science',
            'Physics': 'Science',
            'Chemistry': 'Science',
            'Biology': 'Science',
            'Astronomy': 'Science',
            'Natural science': 'Science',
            # Politics
            'Political fiction': 'Politics',
            'Politics': 'Politics',
            'History': 'Politics',
            'Biography': 'Politics',
            'Political history': 'Politics',
            'Government': 'Politics',
            # Sports
            'Sports': 'Sports',
            'Athletics': 'Sports',
            'Olympic Games': 'Sports',
            # Entertainment
            'Thriller': 'Entertainment',
            'Mystery': 'Entertainment',
            'Romance': 'Entertainment',
            'Drama': 'Entertainment',
            'Adventure': 'Entertainment',
            'Fantasy': 'Entertainment',
            'Horror': 'Entertainment',
            'Comedy': 'Entertainment'
        }

        # assign unified topic based on genres
        def assign_topic(genres):
            for genre in genres:
                if genre in genre_to_topic:
                    return genre_to_topic[genre]
            return None

        df['unified_topic'] = df['genre_list'].apply(assign_topic)

        # keep only mapped topics
        df = df[df['unified_topic'].notna()]

        # rename and validate content
        df = df.rename(columns={'text': 'content'})
        df['valid'] = df['content'].apply(validate_content)
        df = df[df['valid']]

        # sample equally from each topic
        result_df = sample_by_topic(df, samples_per_topic)

        # deduplicate
        result_df = deduplicate_dataframe(result_df)

        result_df['source'] = 'literature'
        result_df = result_df[['content', 'unified_topic', 'source']]

        logger.info(f"✓ Books: {len(result_df)} summaries")
        if len(result_df) > 0:
            logger.info(f"  Topics: {result_df['unified_topic'].value_counts().to_dict()}")

        return result_df

    except requests.RequestException as e:
        logger.error(f"  Network error downloading books: {e}")
        return pd.DataFrame(columns=['content', 'unified_topic', 'source'])
    except Exception as e:
        logger.error(f"  Error processing books: {e}")
        return pd.DataFrame(columns=['content', 'unified_topic', 'source'])
    finally:
        # cleanup temporary directory
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.debug("  Cleaned up temporary files")
            except Exception as e:
                logger.warning(f"  Could not clean up temp dir: {e}")


# download Wikipedia articles by topic using the Wikipedia API
def download_wikipedia_targeted(samples_per_topic: int = SAMPLES_PER_TOPIC_WIKIPEDIA) -> pd.DataFrame:
    logger.info("Downloading Wikipedia articles...")

    # topic-specific search queries
    wiki_queries = {
        'Technology': ['artificial intelligence', 'computer science', 'software engineering', 'machine learning',
                      'programming', 'internet', 'robotics', 'cybersecurity'],
        'Business': ['business', 'company', 'corporation', 'entrepreneurship', 'marketing',
                    'finance', 'stock market', 'economics'],
        'Science': ['physics', 'chemistry', 'biology', 'astronomy', 'geology',
                   'scientific method', 'research', 'laboratory'],
        'Health': ['medicine', 'disease', 'health care', 'medical treatment', 'anatomy',
                  'nutrition', 'mental health', 'public health'],
        'Politics': ['politics', 'government', 'democracy', 'election', 'legislation',
                    'political party', 'international relations', 'diplomacy'],
        'Sports': ['sport', 'olympic games', 'football', 'basketball', 'athletics',
                  'championship', 'athlete', 'tournament'],
        'Entertainment': ['film', 'television', 'music', 'cinema', 'actor',
                         'entertainment industry', 'theater', 'celebrity']
    }

    all_articles = []
    base_url = "https://en.wikipedia.org/w/api.php"

    # Wikipedia required User-Agent header
    headers = {
        'User-Agent': 'TextAnalyticsDataCollector/1.0 (Educational Research Project)'
    }

    for topic, queries in wiki_queries.items():
        logger.info(f"  Fetching {topic} articles...")
        topic_articles = []
        articles_per_query = (samples_per_topic // len(queries)) + 50  # fetch extra for filtering

        for query in tqdm(queries, desc=f"    {topic}", leave=False):
            try:
                # search for articles
                search_params = {
                    'action': 'query',
                    'list': 'search',
                    'srsearch': query,
                    'srlimit': articles_per_query,
                    'format': 'json'
                }

                search_response = requests.get(base_url, params=search_params, headers=headers, timeout=30)
                search_response.raise_for_status()
                search_data = search_response.json()

                if 'query' not in search_data or 'search' not in search_data['query']:
                    continue

                # get content for each article
                for article in search_data['query']['search']:
                    try:
                        content_params = {
                            'action': 'query',
                            'prop': 'extracts',
                            'exintro': True,  # only introduction
                            'explaintext': True,  # plain text
                            'titles': article['title'],
                            'format': 'json'
                        }

                        content_response = requests.get(base_url, params=content_params, headers=headers, timeout=30)
                        content_response.raise_for_status()
                        content_data = content_response.json()

                        pages = content_data.get('query', {}).get('pages', {})
                        for page_id, page_data in pages.items():
                            extract = page_data.get('extract', '')

                            if extract and validate_content(extract):
                                topic_articles.append({
                                    'content': extract,
                                    'unified_topic': topic,
                                    'source': 'encyclopedia'
                                })

                        # rate limiting
                        time.sleep(WIKIPEDIA_REQUEST_DELAY)

                    except Exception as e:
                        logger.debug(f"      Skipped article {article.get('title', 'unknown')}: {e}")
                        continue

            except requests.RequestException as e:
                logger.warning(f"    Wikipedia API error for query '{query}': {e}")
                continue
            except Exception as e:
                logger.warning(f"    Could not process query '{query}': {e}")
                continue

        # sample to target amount
        if len(topic_articles) > samples_per_topic:
            random.seed(RANDOM_SEED)
            topic_articles = random.sample(topic_articles, samples_per_topic)

        all_articles.extend(topic_articles)
        logger.info(f"    Collected {len(topic_articles)} articles")

    df = pd.DataFrame(all_articles)

    if len(df) == 0:
        logger.warning("✗ Wikipedia: 0 articles collected (API may be blocking requests)")
        return pd.DataFrame(columns=['content', 'unified_topic', 'source'])

    # deduplicate
    df = deduplicate_dataframe(df)

    logger.info(f"✓ Wikipedia: {len(df)} articles")
    logger.info(f"  Topics: {df['unified_topic'].value_counts().to_dict()}")

    return df


# download movie reviews
def download_rotten_tomatoes(max_samples: int = SAMPLES_ENTERTAINMENT) -> pd.DataFrame:
    logger.info("Downloading Rotten Tomatoes reviews...")

    try:
        dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")

        # combine all splits
        train_df = pd.DataFrame(dataset['train'])
        val_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # prepare content
        full_df = full_df.rename(columns={'text': 'content'})

        # validate content
        full_df['valid'] = full_df['content'].apply(validate_content)
        full_df = full_df[full_df['valid']].copy()

        # assign unified topic
        full_df['unified_topic'] = 'Entertainment'
        full_df['source'] = 'entertainment'

        # sample to target amount
        n_samples = min(max_samples, len(full_df))
        result_df = full_df.sample(n=n_samples, random_state=RANDOM_SEED)

        # deduplicate
        result_df = deduplicate_dataframe(result_df)

        result_df = result_df[['content', 'unified_topic', 'source']]

        logger.info(f"✓ Rotten Tomatoes: {len(result_df)} reviews")

        return result_df

    except Exception as e:
        logger.error(f"  Error downloading Rotten Tomatoes: {e}")
        return pd.DataFrame(columns=['content', 'unified_topic', 'source'])


# download all and create unified dataset
def main():
    logger.info("=" * 60)
    logger.info("Content Classifier: Unified Data Import")
    logger.info("=" * 60)
    logger.info("\nTarget structure:")
    logger.info(f"  News (4 topics): {SAMPLES_PER_TOPIC_NEWS} per topic")
    logger.info(f"  Scientific (4 topics): {SAMPLES_PER_TOPIC_ARXIV} per topic")
    logger.info(f"  Literature (varies): {SAMPLES_PER_TOPIC_BOOKS} per topic")
    logger.info(f"  Wikipedia (7 topics): {SAMPLES_PER_TOPIC_WIKIPEDIA} per topic")
    logger.info(f"  Entertainment: {SAMPLES_ENTERTAINMENT} samples")
    logger.info(f"  Total topics: Technology, Business, Science, Health, Politics, Sports, Entertainment\n")

    # check dependencies first
    try:
        check_dependencies()
    except ImportError as e:
        logger.error(f"Dependency check failed: {e}")
        return

    # download each dataset
    news_df = download_ag_news_targeted()
    arxiv_df = download_arxiv_targeted()
    books_df = download_books_targeted()
    wikipedia_df = download_wikipedia_targeted()
    entertainment_df = download_rotten_tomatoes()

    # combine all datasets
    combined_df = pd.concat([news_df, arxiv_df, books_df, wikipedia_df, entertainment_df], ignore_index=True)

    # deduplicate across all sources
    logger.info("\nDeduplicating across all sources...")
    combined_df = deduplicate_dataframe(combined_df)

    # save individual sources
    logger.info("\nSaving datasets...")
    news_df.to_csv(DATA_DIR / "news_processed.csv", index=False)
    arxiv_df.to_csv(DATA_DIR / "arxiv_processed.csv", index=False)
    if len(books_df) > 0:
        books_df.to_csv(DATA_DIR / "books_processed.csv", index=False)
    if len(wikipedia_df) > 0:
        wikipedia_df.to_csv(DATA_DIR / "wikipedia_processed.csv", index=False)
    if len(entertainment_df) > 0:
        entertainment_df.to_csv(DATA_DIR / "entertainment_processed.csv", index=False)

    # save combined dataset
    combined_df.to_csv(DATA_DIR / "combined_dataset.csv", index=False)

    logger.info("\n" + "=" * 60)
    logger.info("Download Complete!")
    logger.info("=" * 60)

    # summary statistics
    logger.info(f"\nCombined Dataset Summary:")
    logger.info(f"  Total samples: {len(combined_df):,}")
    logger.info(f"\nBy source:")
    logger.info(f"\n{combined_df['source'].value_counts()}")
    logger.info(f"\nBy topic:")
    logger.info(f"\n{combined_df['unified_topic'].value_counts()}")
    logger.info(f"\nBy topic and source:")
    logger.info(f"\n{combined_df.groupby(['unified_topic', 'source']).size()}")

    # text length statistics
    combined_df['word_count'] = combined_df['content'].str.split().str.len()
    logger.info(f"\nText length statistics:")
    logger.info(f"  Mean: {combined_df['word_count'].mean():.1f} words")
    logger.info(f"  Median: {combined_df['word_count'].median():.1f} words")
    logger.info(f"  Min: {combined_df['word_count'].min()} words")
    logger.info(f"  Max: {combined_df['word_count'].max()} words")

    logger.info(f"\nFiles saved in: {DATA_DIR}/")
    logger.info("  - news_processed.csv")
    logger.info("  - arxiv_processed.csv")
    logger.info("  - books_processed.csv (if available)")
    logger.info("  - wikipedia_processed.csv")
    logger.info("  - entertainment_processed.csv")
    logger.info("  - combined_dataset.csv (main file)")

    logger.info("\nNext step: Run notebooks/01_dataset_eda.ipynb for detailed analysis")


if __name__ == "__main__":
    main()
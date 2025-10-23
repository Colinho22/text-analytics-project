"""
Data splitting utilities for creating reproducible train/validation/test splits.

This module ensures all methods use the same data splits for fair comparison.
Splits are stratified to maintain class balance across all sets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
import os

# Set random seed for reproducibility
RANDOM_STATE = 42


def create_stratified_splits(
        df: pd.DataFrame,
        text_column: str = 'content',
        label_column: str = 'unified_topic',
        train_size: float = 0.70,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation/test splits from a DataFrame.

    Args:
        df: Input DataFrame with text and labels
        text_column: Name of the text column
        label_column: Name of the label column
        train_size: Proportion for training set (default 0.70)
        val_size: Proportion for validation set (default 0.15)
        test_size: Proportion for test set (default 0.15)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Validate split sizes
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(f"Split sizes must sum to 1.0, got {train_size + val_size + test_size}")

    # Ensure required columns exist
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame")

    # First split: separate out test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_column],
        random_state=random_state
    )

    # Second split: separate train and validation from remaining data
    # Adjust validation size relative to the remaining data
    val_size_adjusted = val_size / (train_size + val_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        stratify=train_val_df[label_column],
        random_state=random_state
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def calculate_split_statistics(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label_column: str = 'unified_topic'
) -> Dict:
    """
    Calculate statistics about the data splits.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        label_column: Name of the label column

    Returns:
        Dictionary with split statistics
    """
    stats = {
        'total_samples': len(train_df) + len(val_df) + len(test_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'train_percentage': len(train_df) / (len(train_df) + len(val_df) + len(test_df)) * 100,
        'val_percentage': len(val_df) / (len(train_df) + len(val_df) + len(test_df)) * 100,
        'test_percentage': len(test_df) / (len(train_df) + len(val_df) + len(test_df)) * 100,
        'num_classes': train_df[label_column].nunique(),
        'class_names': sorted(train_df[label_column].unique().tolist())
    }

    # Calculate per-class distribution for each split
    stats['train_class_distribution'] = train_df[label_column].value_counts().to_dict()
    stats['val_class_distribution'] = val_df[label_column].value_counts().to_dict()
    stats['test_class_distribution'] = test_df[label_column].value_counts().to_dict()

    return stats


def save_splits(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str = None,
        save_stats: bool = True
) -> None:
    """
    Save train/validation/test splits to CSV files.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Directory to save the splits (default: data/processed from project root)
        save_stats: Whether to save statistics file
    """
    # Use default path relative to project root if not specified
    if output_dir is None:
        # Try to find project root
        current_file = Path(__file__).resolve()
        if current_file.parent.name == 'src':
            project_root = current_file.parent.parent
        else:
            project_root = current_file.parent
        output_path = project_root / 'data' / 'processed'
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save splits to CSV
    train_path = output_path / 'train.csv'
    val_path = output_path / 'val.csv'
    test_path = output_path / 'test.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✅ Saved train split to {train_path} ({len(train_df)} samples)")
    print(f"✅ Saved validation split to {val_path} ({len(val_df)} samples)")
    print(f"✅ Saved test split to {test_path} ({len(test_df)} samples)")

    # Save statistics if requested
    if save_stats:
        stats = calculate_split_statistics(train_df, val_df, test_df)
        stats_path = output_path / 'split_statistics.txt'

        with open(stats_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DATA SPLIT STATISTICS\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total samples: {stats['total_samples']:,}\n")
            f.write(f"Training:      {stats['train_samples']:,} ({stats['train_percentage']:.1f}%)\n")
            f.write(f"Validation:    {stats['val_samples']:,} ({stats['val_percentage']:.1f}%)\n")
            f.write(f"Test:          {stats['test_samples']:,} ({stats['test_percentage']:.1f}%)\n\n")

            f.write(f"Number of classes: {stats['num_classes']}\n")
            f.write(f"Classes: {', '.join(stats['class_names'])}\n\n")

            f.write("=" * 60 + "\n")
            f.write("CLASS DISTRIBUTION PER SPLIT\n")
            f.write("=" * 60 + "\n\n")

            # Training distribution
            f.write("Training Set:\n")
            for class_name in sorted(stats['train_class_distribution'].keys()):
                count = stats['train_class_distribution'][class_name]
                percentage = count / stats['train_samples'] * 100
                f.write(f"  {class_name:15s}: {count:6,} ({percentage:5.2f}%)\n")

            f.write("\nValidation Set:\n")
            for class_name in sorted(stats['val_class_distribution'].keys()):
                count = stats['val_class_distribution'][class_name]
                percentage = count / stats['val_samples'] * 100
                f.write(f"  {class_name:15s}: {count:6,} ({percentage:5.2f}%)\n")

            f.write("\nTest Set:\n")
            for class_name in sorted(stats['test_class_distribution'].keys()):
                count = stats['test_class_distribution'][class_name]
                percentage = count / stats['test_samples'] * 100
                f.write(f"  {class_name:15s}: {count:6,} ({percentage:5.2f}%)\n")

        print(f"✅ Saved statistics to {stats_path}")


def load_splits(
        data_dir: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load previously saved train/validation/test splits.

    Args:
        data_dir: Directory containing the split files (default: data/processed from project root)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Use default path relative to project root if not specified
    if data_dir is None:
        # Try to find project root
        current_file = Path(__file__).resolve()
        if current_file.parent.name == 'src':
            project_root = current_file.parent.parent
        else:
            project_root = current_file.parent
        data_path = project_root / 'data' / 'processed'
    else:
        data_path = Path(data_dir)

    train_path = data_path / 'train.csv'
    val_path = data_path / 'val.csv'
    test_path = data_path / 'test.csv'

    # Check if files exist
    if not train_path.exists():
        raise FileNotFoundError(f"Training split not found at {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation split not found at {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found at {test_path}")

    # Load splits
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"✅ Loaded train split: {len(train_df):,} samples")
    print(f"✅ Loaded validation split: {len(val_df):,} samples")
    print(f"✅ Loaded test split: {len(test_df):,} samples")

    return train_df, val_df, test_df


def verify_split_integrity(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_column: str = 'content'
) -> bool:
    """
    Verify that splits have no overlapping samples.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        text_column: Name of the text column to check

    Returns:
        True if splits are valid (no overlap), False otherwise
    """
    train_texts = set(train_df[text_column])
    val_texts = set(val_df[text_column])
    test_texts = set(test_df[text_column])

    # Check for overlaps
    train_val_overlap = train_texts & val_texts
    train_test_overlap = train_texts & test_texts
    val_test_overlap = val_texts & test_texts

    if train_val_overlap:
        print(f"❌ Warning: {len(train_val_overlap)} samples overlap between train and validation")
        return False

    if train_test_overlap:
        print(f"❌ Warning: {len(train_test_overlap)} samples overlap between train and test")
        return False

    if val_test_overlap:
        print(f"❌ Warning: {len(val_test_overlap)} samples overlap between validation and test")
        return False

    print("✅ All splits are valid - no overlapping samples")
    return True


if __name__ == "__main__":
    # Example usage and testing

    print("Data Splitting Module - Test Run")
    print("=" * 60)

    # Determine project root (works whether running from root or src/)
    current_file = Path(__file__).resolve()
    if current_file.parent.name == 'src':
        # Running from src/ directory
        project_root = current_file.parent.parent
    else:
        # Running from project root
        project_root = current_file.parent

    # Construct path to data file
    data_file = project_root / 'data' / 'combined_dataset.csv'

    if not data_file.exists():
        print(f"\n⚠️  Dataset not found at {data_file}")
        print("Please run data_download.py first to create the dataset.")
        print("\nFor testing purposes, here's what would happen:")
        print("\n1. Load data/combined_dataset.csv")
        print("2. Create stratified 70/15/15 splits")
        print("3. Save to data/processed/train.csv, val.csv, test.csv")
        print("4. Generate statistics file")
        print("5. Verify no overlapping samples")
    else:
        print(f"\n✅ Found dataset at {data_file}")
        print("\nLoading data...")

        df = pd.read_csv(str(data_file))
        print(f"Total samples: {len(df):,}")

        print("\nCreating stratified splits (70/15/15)...")
        train_df, val_df, test_df = create_stratified_splits(df)

        print("\nSaving splits...")
        save_splits(train_df, val_df, test_df)

        print("\nVerifying split integrity...")
        verify_split_integrity(train_df, val_df, test_df)

        print("\n" + "=" * 60)
        print("✅ Data splitting complete!")
        print("=" * 60)
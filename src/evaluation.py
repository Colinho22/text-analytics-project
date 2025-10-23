# evaluation utilities for text classification models.
# This module provides consistent evaluation metrics and visualizations
# that will be used by all methods for fair comparison.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import json

# set plotting style
plt.style.use('default')
sns.set_palette("Set2")


def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None
) -> Dict:

    # calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # calculate per-class and macro-averaged metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )

    # calculate macro averages
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)

    # get unique classes
    if class_names is None:
        class_names = [str(c) for c in np.unique(np.concatenate([y_true, y_pred]))]

    # build per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        if i < len(precision):
            per_class_metrics[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1-score': float(f1[i]),
                'f1': float(f1[i]),  # Alias for backwards compatibility
                'support': int(support[i])
            }

    # compile all metrics
    metrics = {
        'accuracy': float(accuracy),
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'macro_f1': float(f1_macro),
        'per_class_metrics': per_class_metrics,  # Fixed key name
        'num_samples': len(y_true)
    }

    return metrics


def generate_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: bool = False
) -> np.ndarray:

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        # normalize by row (true label)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm


def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        output_path: Optional[str] = None,
        normalize: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        title: str = "Confusion Matrix"
) -> plt.Figure:

    # generate confusion matrix
    cm = generate_confusion_matrix(y_true, y_pred, class_names, normalize=normalize)

    # create figure
    fig, ax = plt.subplots(figsize=figsize)

    # plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='YlGnBu',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        ax=ax
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)

    # rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    # save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved confusion matrix to {output_path}")

    return fig


def get_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        output_path: Optional[str] = None
) -> str:

    # generate report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    # save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        print(f"✅ Saved classification report to {output_path}")

    return report


def find_most_confused_pairs(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        top_n: int = 5
) -> List[Tuple[str, str, int]]:
    # find the most commonly confused class pairs.

    # generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # find off-diagonal elements (misclassifications)
    confused_pairs = []
    n_classes = len(class_names)

    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((
                    class_names[i],
                    class_names[j],
                    int(cm[i, j])
                ))

    # sort by count (descending)
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    return confused_pairs[:top_n]


def save_results(
        metrics: Dict,
        method_name: str,
        output_dir: str = None
) -> None:
    # save evaluation results to JSON file.

    # determine output directory
    if output_dir is None:
        # auto-detect project root
        current_file = Path(__file__).resolve()
        if current_file.parent.name == 'src':
            project_root = current_file.parent.parent
        else:
            project_root = current_file.parent
        output_path = project_root / 'results'
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # save metrics as JSON
    results_file = output_path / f'{method_name}_metrics.json'
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Saved metrics to {results_file}")


def evaluate_model(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        method_name: str,
        output_dir: str = None,
        save_plots: bool = True,
        training_time_seconds: float = None
) -> Dict:
    # complete evaluation pipeline: calculate metrics, generate plots, save results.
    print(f"\n{'=' * 60}")
    print(f"EVALUATING: {method_name.upper()}")
    print(f"{'=' * 60}\n")

    # calculate metrics
    metrics = calculate_metrics(y_true, y_pred, class_names)

    # add training time if provided
    if training_time_seconds is not None:
        metrics['training_time_seconds'] = float(training_time_seconds)

    # print summary
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"Samples:         {metrics['num_samples']:,}")

    # determine output directory
    if output_dir is None:
        current_file = Path(__file__).resolve()
        if current_file.parent.name == 'src':
            project_root = current_file.parent.parent
        else:
            project_root = current_file.parent
        output_path = project_root / 'results'
    else:
        output_path = Path(output_dir)

    # save results
    save_results(metrics, method_name, output_path)

    # generate and save plots
    if save_plots:
        # confusion matrix
        cm_path = output_path / f'{method_name}_confusion_matrix.png'
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            output_path=str(cm_path),
            title=f'Confusion Matrix - {method_name.upper()}'
        )
        plt.close()

        # classification report
        report_path = output_path / f'{method_name}_classification_report.txt'
        get_classification_report(y_true, y_pred, class_names, output_path=str(report_path))

        # find most confused pairs
        confused = find_most_confused_pairs(y_true, y_pred, class_names)
        print(f"\nMost confused class pairs:")
        for true_class, pred_class, count in confused[:5]:
            print(f"  {true_class} → {pred_class}: {count} samples")

    print(f"\n{'=' * 60}\n")

    return metrics


if __name__ == "__main__":
    # test the evaluation functions
    print("Evaluation Module - Test Run")
    print("=" * 60)

    # create synthetic test data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 7

    class_names = [
        'Business', 'Entertainment', 'Health',
        'Politics', 'Science', 'Sports', 'Technology'
    ]

    # simulate predictions with some errors
    y_true = np.random.randint(0, n_classes, n_samples)

    # create predictions that are mostly correct
    y_pred = y_true.copy()

    # add some errors
    error_indices = np.random.choice(n_samples, size=200, replace=False)
    y_pred[error_indices] = np.random.randint(0, n_classes, 200)

    print(f"\nTesting with {n_samples} samples, {n_classes} classes")
    print(f"Simulated accuracy: ~{((y_true == y_pred).sum() / n_samples * 100):.1f}%")

    # test evaluation pipeline
    metrics = evaluate_model(
        y_true,
        y_pred,
        class_names,
        method_name='test_model',
        save_plots=True
    )

    print("\n" + "=" * 60)
    print("✅ Evaluation module test complete!")
    print("=" * 60)
    print("\nCheck results/ directory for generated files:")
    print("  • test_model_metrics.json")
    print("  • test_model_confusion_matrix.png")
    print("  • test_model_classification_report.txt")
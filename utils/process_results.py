import os

import pandas                as pd
import numpy                 as np
import matplotlib.pyplot     as plt

from   sklearn.metrics       import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from   sklearn.preprocessing import LabelEncoder, label_binarize

def process_metrics(csv_path: str):
    df        = pd.read_csv(csv_path)
    y_true    = df['True_Label'].astype(int)
    y_pred    = df['Predicted_Label'].astype(int)
    y_score   = df['Prediction_Probability']

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    np.random.seed(42)
    n_bootstraps     = 100
    auc_scores       = []
    accuracy_scores  = []
    precision_scores = []
    recall_scores    = []
    f1_scores        = []

    fpr_list         = []
    tpr_list         = []

    for _ in range(n_bootstraps):
        indices      = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot  = y_true.iloc[indices]
        y_pred_boot  = y_pred.iloc[indices]
        y_score_boot = y_score.iloc[indices]

        auc_scores.append(roc_auc_score(y_true_boot, y_score_boot))
        accuracy_scores.append(accuracy_score(y_true_boot, y_pred_boot))
        precision_scores.append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
        recall_scores.append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
        f1_scores.append(f1_score(y_true_boot, y_pred_boot, zero_division=0))

        fpr, tpr, _ = roc_curve(y_true_boot, y_score_boot)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    mean_auc     = np.mean(auc_scores)
    ci_auc       = (np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))
    ci_accuracy  = (np.percentile(accuracy_scores, 2.5), np.percentile(accuracy_scores, 97.5))
    ci_precision = (np.percentile(precision_scores, 2.5), np.percentile(precision_scores, 97.5))
    ci_recall    = (np.percentile(recall_scores, 2.5), np.percentile(recall_scores, 97.5))
    ci_f1        = (np.percentile(f1_scores, 2.5), np.percentile(f1_scores, 97.5))

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)], axis=0)
    std_tpr  = np.std([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)], axis=0)

    roc_data = {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'std_tpr': std_tpr
    }

    return {
        'accuracy': (accuracy, ci_accuracy),
        'precision': (precision, ci_precision),
        'recall': (recall, ci_recall),
        'f1_score': (f1, ci_f1),
        'auc_mean': mean_auc,
        'auc_ci': ci_auc,
        'roc_data': roc_data
    }


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compare_models(csv_paths: list, names: list):
    assert len(csv_paths) == len(names), "Mismatch between number of paths and names"
    results = []
    plt.figure(figsize=(10, 6))

    for csv_path, name in zip(csv_paths, names):
        metrics  = process_metrics(csv_path)
        roc      = metrics.pop('roc_data')
        mean_fpr = roc['mean_fpr']
        mean_tpr = roc['mean_tpr']
        std_tpr  = roc['std_tpr']

        mean_fpr = np.concatenate(([0.], mean_fpr, [1.]))
        mean_tpr = np.concatenate(([0.], mean_tpr, [1.]))
        std_tpr  = np.concatenate(([0.], std_tpr,  [0.]))

        plt.plot(mean_fpr, mean_tpr,
                 label=f"{name} (AUC: {metrics['auc_mean']:.3f})")
        plt.fill_between(mean_fpr,
                         mean_tpr - std_tpr,
                         mean_tpr + std_tpr,
                         alpha=0.2)

        results.append({
            "Model"    : name,
            "Accuracy" : f"{metrics['accuracy'][0]:.3f} ({metrics['accuracy'][1][0]:.3f}-{metrics['accuracy'][1][1]:.3f})",
            "Precision": f"{metrics['precision'][0]:.3f} ({metrics['precision'][1][0]:.3f}-{metrics['precision'][1][1]:.3f})",
            "Recall"   : f"{metrics['recall'][0]:.3f} ({metrics['recall'][1][0]:.3f}-{metrics['recall'][1][1]:.3f})",
            "F1 Score" : f"{metrics['f1_score'][0]:.3f} ({metrics['f1_score'][1][0]:.3f}-{metrics['f1_score'][1][1]:.3f})",
            "AUC"      : f"{metrics['auc_mean']:.3f} ({metrics['auc_ci'][0]:.3f}-{metrics['auc_ci'][1]:.3f})",
        })

    plt.plot([0, 1], [0, 1],
             linestyle='--',
             color='black',
             label='Random (AUC = 0.50)')

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    df_results = pd.DataFrame(results)
    print("\nEvaluation Metrics:")
    print(df_results.to_markdown(index=False))


def plot_multiclass(csv_path, title=None):
    """
    Load a CSV file, automatically detect multiclass structure, and plot ROC curves per class.

    Parameters:
    - csv_path (str): Path to the CSV file.
    - title (str): Optional custom title for the plot. If None, uses the CSV filename.
    """
    df = pd.read_csv(csv_path)
    
    if title is None:
        base  = os.path.basename(csv_path)
        title = f"Multiclass AUC - {os.path.splitext(base)[0]}"

    label_col = next((col for col in ['True_Label', 'true_label', 'label', 'Target'] if col in df.columns), None)

    if label_col is None:
        raise ValueError("Could not find a true label column.")

    prob_cols = [
        col for col in df.columns

        if pd.api.types.is_float_dtype(df[col]) and df[col].between(0, 1).all()
    ]

    if label_col in prob_cols:
        prob_cols.remove(label_col)

    class_names = [col.replace("prob_", "").replace("Prob_Class_", "") for col in prob_cols]

    le          = LabelEncoder()
    y_true      = le.fit_transform(df[label_col])
    y_true_bin  = label_binarize(y_true, classes=range(len(class_names)))
    y_scores    = df[prob_cols].values

    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i]        = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))

    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\nAUC Scores for {title}")

    for cname, score in zip(class_names, roc_auc.values()):
        print(f"{cname:10s} : {score:.3f}")
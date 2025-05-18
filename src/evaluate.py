import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, matthews_corrcoef
import matplotlib.pyplot as plt
from utils import cosine_similarity, load_embedding, ensure_dir

def evaluate_pairs(pairs, output_dir, model_name="arcface"):
    y_true, y_score = [], []
    for f1, f2, label in pairs:
        emb1, emb2 = load_embedding(f1), load_embedding(f2)
        score = cosine_similarity(emb1, emb2)
        y_true.append(label)
        y_score.append(score)
    y_true, y_score = np.array(y_true), np.array(y_score)

    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # EER calculation
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    # TPR@FAR=1e-3, 1e-4
    def tpr_at_fpr(target_fpr):
        idx = np.where(fpr <= target_fpr)[0]
        return tpr[idx[-1]] if len(idx) else 0.0

    tpr_at_far_1e3 = tpr_at_fpr(1e-3)
    tpr_at_far_1e4 = tpr_at_fpr(1e-4)

    # Best threshold (Youden’s J stat)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_thresh = thresholds[optimal_idx]
    y_pred = (y_score >= optimal_thresh).astype(int)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Print & Save
    print(f"--- {model_name.upper()} BENCHMARK ---")
    print(f"Optimal threshold: {optimal_thresh:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"EER: {eer:.4f} (threshold={eer_threshold:.4f})")
    print(f"TPR@FAR=1e-3: {tpr_at_far_1e3:.4f}")
    print(f"TPR@FAR=1e-4: {tpr_at_far_1e4:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    ensure_dir(f"{output_dir}/plots/")
    # Confusion matrix plot
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.savefig(f"{output_dir}/plots/cm_{model_name}.png")
    plt.close()

    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"{output_dir}/plots/roc_{model_name}.png")
    plt.close()

    # EER plot
    plt.figure()
    plt.plot(fpr, fnr, label='FNR vs FPR')
    plt.plot(fpr[eer_idx], fnr[eer_idx], 'ro', label=f'EER={eer:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title(f'{model_name} EER')
    plt.legend()
    plt.savefig(f"{output_dir}/plots/eer_{model_name}.png")
    plt.close()

    # Save metrics to text file
    with open(f"{output_dir}/{model_name}_metrics.txt", 'w') as f:
        f.write(f"Threshold: {optimal_thresh:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\nF1: {f1:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\nAUC: {roc_auc:.4f}\n")
        f.write(f"EER: {eer:.4f} (at threshold {eer_threshold:.4f})\n")
        f.write(f"TPR@FAR=1e-3: {tpr_at_far_1e3:.4f}\n")
        f.write(f"TPR@FAR=1e-4: {tpr_at_far_1e4:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
    return y_true, y_pred, y_score, optimal_thresh, cm


def get_misclassified_pairs(pairs, y_true, y_pred, max_examples=20):
    errors = []
    for i, (label, pred) in enumerate(zip(y_true, y_pred)):
        if label != pred:
            errors.append(pairs[i])
    return errors[:max_examples]

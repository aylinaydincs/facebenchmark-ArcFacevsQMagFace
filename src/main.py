import os
import random
import numpy as np
import argparse
from utils import save_lfw_sklearn, generate_random_pairs, parse_lfw_pairs_csv, generate_sklearn_pairs
from evaluate import evaluate_pairs, get_misclassified_pairs
from visualize import save_misclassified_samples
from src.embeddings import extract_arcface_embeddings_deepface, extract_qmagface_embeddings

class Config:
    @staticmethod
    def from_args():
        parser = argparse.ArgumentParser(description="Face Recognition Benchmark Pipeline")
        parser.add_argument('--arcface_emb_dir', type=str, default='results/arcface_embeddings', help='ArcFace embeddings directory')
        parser.add_argument('--qmagface_emb_dir', type=str, default='results/qmagface_embeddings', help='QMagFace embeddings directory')
        parser.add_argument('--lfw_pairs_path', type=str, default='data/pairs.csv', help='LFW pairs CSV path')
        parser.add_argument('--results_dir', type=str, default='results/benchmark_sklearn', help='Results directory')
        parser.add_argument('--img_root', type=str, default='data/lfw_sklearn', help='Image root directory')
        parser.add_argument('--qmag_model', type=str, default='models/magface_model.pth', help='QMagFace model path')
        parser.add_argument('--n_pairs', type=int, default=3000, help='Number of pairs for sklearn pairs')
        parser.add_argument('--seed', type=int, default=42, help='Random seed')
        return parser.parse_args()

def ensure_dataset(img_root):
    if not (os.path.exists(img_root) and os.listdir(img_root)):
        print("LFW dataset not found. Downloading with scikit-learn...")
        save_lfw_sklearn(data_dir=img_root, min_faces_per_person=2, resize=1.0, color=True)

def ensure_embeddings(cfg):
    if not (os.path.exists(cfg.arcface_emb_dir) and os.listdir(cfg.arcface_emb_dir)):
        print("Extracting ArcFace embeddings...")
        extract_arcface_embeddings_deepface(cfg.img_root, cfg.arcface_emb_dir)
    if not (os.path.exists(cfg.qmagface_emb_dir) and os.listdir(cfg.qmagface_emb_dir)):
        print("Extracting QMagFace embeddings...")
        extract_qmagface_embeddings(cfg.img_root, cfg.qmagface_emb_dir, cfg.qmag_model)

def main():
    cfg = Config.from_args()

    # Set seeds for reproducibility
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Ensure dataset exists
    ensure_dataset(cfg.img_root)

    # Ensure embeddings exist
    ensure_embeddings(cfg)

    # Generate pairs (optionally use parse_lfw_pairs_csv for official LFW pairs)
    # arc_pairs = generate_random_pairs(cfg.arcface_emb_dir, n_pairs=cfg.n_pairs, seed=cfg.seed)
    # qmag_pairs = generate_random_pairs(cfg.qmagface_emb_dir, n_pairs=cfg.n_pairs, seed=cfg.seed)
    # arc_pairs = parse_lfw_pairs_csv(cfg.lfw_pairs_path, cfg.arcface_emb_dir)
    # qmag_pairs = parse_lfw_pairs_csv(cfg.lfw_pairs_path, cfg.qmagface_emb_dir)

    arc_pairs = generate_sklearn_pairs(cfg.arcface_emb_dir, n_pairs=cfg.n_pairs)
    qmag_pairs = generate_sklearn_pairs(cfg.qmagface_emb_dir, n_pairs=cfg.n_pairs)

    # Evaluate ArcFace
    y_true_arc, y_pred_arc, y_score_arc, arc_thresh, cm_arc = evaluate_pairs(
        arc_pairs, cfg.results_dir, model_name='arcface'
    )
    misclassified_arc = get_misclassified_pairs(arc_pairs, y_true_arc, y_pred_arc)
    save_misclassified_samples(misclassified_arc, os.path.join(cfg.results_dir, 'arcface/misclassified'))

    # Evaluate QMagFace
    y_true_qmag, y_pred_qmag, y_score_qmag, qmag_thresh, cm_qmag = evaluate_pairs(
        qmag_pairs, cfg.results_dir, model_name='qmagface'
    )
    misclassified_qmag = get_misclassified_pairs(qmag_pairs, y_true_qmag, y_pred_qmag)
    save_misclassified_samples(misclassified_qmag, os.path.join(cfg.results_dir, 'qmagface/misclassified'))

if __name__ == "__main__":
    main()

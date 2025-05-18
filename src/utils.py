import os
import random
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.datasets import fetch_lfw_people

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_embedding(emb_path):
    return np.load(emb_path)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_image(image_path):
    return Image.open(image_path)

def save_lfw_sklearn(data_dir="data/lfw_sklearn", min_faces_per_person=2, resize=1.0, color=True):
    os.makedirs(data_dir, exist_ok=True)
    print("Downloading LFW using scikit-learn...")
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize, color=color, download_if_missing=True)
    for idx, (img_arr, label) in enumerate(zip(lfw_people.images, lfw_people.target)):
        person = lfw_people.target_names[label]
        person_dir = os.path.join(data_dir, person)
        os.makedirs(person_dir, exist_ok=True)
        # FIX: convert [0,1] floats to uint8 [0,255]
        img = Image.fromarray((img_arr * 255).astype(np.uint8))
        img_path = os.path.join(person_dir, f"{person}_{idx+1:04d}.jpg")
        img.save(img_path)
    print(f"Saved {len(lfw_people.images)} images under {data_dir}")

def list_embeddings(emb_dir):
    all_files = []
    for identity in os.listdir(emb_dir):
        id_dir = os.path.join(emb_dir, identity)
        for fname in os.listdir(id_dir):
            if fname.endswith('.npy'):
                all_files.append((identity, os.path.join(id_dir, fname)))
    return all_files


def generate_sklearn_pairs(emb_dir, n_pairs=3000, seed=42):
    random.seed(seed)
    # Gather all embeddings grouped by identity
    id_to_files = {}
    for identity in os.listdir(emb_dir):
        id_dir = os.path.join(emb_dir, identity)
        if not os.path.isdir(id_dir):
            continue
        files = sorted(glob(os.path.join(id_dir, '*.npy')))
        if files:
            id_to_files[identity] = files
    identities = list(id_to_files.keys())
    # Positive pairs
    positive_pairs = []
    for identity, files in id_to_files.items():
        if len(files) < 2:
            continue
        # All possible unique pairs (or sample if too many)
        pairs = [(files[i], files[j]) for i in range(len(files)) for j in range(i+1, len(files))]
        if len(pairs) > n_pairs:
            pairs = random.sample(pairs, n_pairs)
        for (f1, f2) in pairs:
            positive_pairs.append((f1, f2, 1))
    # Negative pairs
    n_neg = len(positive_pairs)
    negative_pairs = []
    while len(negative_pairs) < n_neg:
        id1, id2 = random.sample(identities, 2)
        f1 = random.choice(id_to_files[id1])
        f2 = random.choice(id_to_files[id2])
        negative_pairs.append((f1, f2, 0))
    print(f"Generated {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs.")
    return positive_pairs + negative_pairs


def generate_random_pairs(emb_dir, n_pairs=3000, seed=42):
    random.seed(seed)
    all_embeddings = list_embeddings(emb_dir)
    id_to_files = {}
    for identity, path in all_embeddings:
        id_to_files.setdefault(identity, []).append(path)
    identities = list(id_to_files.keys())

    # Positive pairs
    positive_pairs = []
    for identity, files in id_to_files.items():
        if len(files) < 2:
            continue
        for _ in range(min(n_pairs, len(files) * (len(files)-1) // 2)):
            img1, img2 = random.sample(files, 2)
            positive_pairs.append((img1, img2, 1))
    # Negative pairs
    negative_pairs = []
    while len(negative_pairs) < len(positive_pairs):
        id1, id2 = random.sample(identities, 2)
        img1 = random.choice(id_to_files[id1])
        img2 = random.choice(id_to_files[id2])
        negative_pairs.append((img1, img2, 0))
    return positive_pairs + negative_pairs


def parse_lfw_pairs_csv(csv_path, emb_dir):
    import pandas as pd
    df = pd.read_csv(csv_path)
    pairs = []
    for idx, row in df.iterrows():
        if pd.isnull(row['Unnamed: 3']):
            # Positive pair
            name = row['name']
            idx1 = int(row['imagenum1'])
            idx2 = int(row['imagenum2'])
            f1 = os.path.join(emb_dir, name, f"{name}_{idx1:04d}.npy")
            f2 = os.path.join(emb_dir, name, f"{name}_{idx2:04d}.npy")
            label = 1
        else:
            # Negative pair
            name1 = row['name']
            idx1 = int(row['imagenum1'])
            name2 = row['imagenum2']
            idx2 = int(row['Unnamed: 3'])
            f1 = os.path.join(emb_dir, name1, f"{name1}_{idx1:04d}.npy")
            f2 = os.path.join(emb_dir, name2, f"{name2}_{idx2:04d}.npy")
            label = 0
        if os.path.exists(f1) and os.path.exists(f2):
            pairs.append((f1, f2, label))
        else:
            print(f"Warning: Missing embedding: {f1} or {f2}")
    return pairs

import os
import numpy as np
from tqdm import tqdm
import cv2
import torch
import torchvision.transforms
from torch.utils import data
from collections import namedtuple
from utils import ensure_dir
from QMagFace.utils.files import list_all_files
from QMagFace.preprocessing.magface.network_inf import builder_inf
from deepface import DeepFace
import argparse


class ImgDataset(data.Dataset):
    def __init__(self, filenames, transform=None):
        super().__init__()
        self.filenames = filenames
        self.transform = transform

    def __getitem__(self, item):
        path = self.filenames[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        if self.transform:
            img = self.transform(img)
        return img, path

    def __len__(self):
        return len(self.filenames)

# ------------------ ArcFace Embedding Extraction ------------------
def extract_arcface_embeddings_deepface(input_root, output_root, model_name='ArcFace', batch_size=64):
    """
    Extract embeddings using DeepFace ArcFace model with DataLoader.
    """
    print("Loading DeepFace ArcFace model...")
    model = DeepFace.build_model(model_name)
    print("Model loaded.")

    # Collect all image paths and their output paths
    img_paths = []
    out_paths = []
    for identity in os.listdir(input_root):
        id_folder = os.path.join(input_root, identity)
        out_id_folder = os.path.join(output_root, identity)
        ensure_dir(out_id_folder)
        for img_file in os.listdir(id_folder):
            img_path = os.path.join(id_folder, img_file)
            out_file = os.path.join(out_id_folder, os.path.splitext(img_file)[0] + '.npy')
            img_paths.append(img_path)
            out_paths.append(out_file)

    dataset = ImgDataset(img_paths)
    loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    for batch_imgs, batch_paths in tqdm(loader, desc='ArcFace: Embedding'):
        # DeepFace expects image paths, so we use batch_paths
        for img_path, out_file in zip(batch_paths, out_paths):
            embedding = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=False)
            if isinstance(embedding, list) and isinstance(embedding[0], dict):
                emb = np.array(embedding[0]['embedding'])
            else:
                emb = np.array(embedding)
            np.save(out_file, emb)
    print(f"ArcFace: All embeddings are saved under {output_root}")

# ------------------ QMagFace Embedding Extraction ------------------
def extract_qmagface_embeddings(source_dir, result_dir, model_path='models/magface_model.pth', batch_size=64):
    """
    Extract embeddings using QMagFace model.
    """
    def save_embedding_for_path(embedding, img_path, source_root, result_root):
        rel_path = os.path.relpath(img_path, start=source_root)
        rel_dir = os.path.dirname(rel_path)
        base, _ = os.path.splitext(os.path.basename(img_path))
        save_dir = os.path.join(result_root, rel_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{base}.npy")
        np.save(save_path, embedding)

    os.makedirs(result_dir, exist_ok=True)
    Args = namedtuple('Args', ['arch', 'resume', 'embedding_size', 'cpu_mode'])
    args = Args('iresnet100', model_path, 512, True)
    model = builder_inf(args)
    model = torch.nn.DataParallel(model)
    model.eval()

    # Try GPU first, fallback to CPU if not available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    trans = torchvision.transforms.ToTensor()
    filenames = list_all_files(source_dir)
    print(f'QMagFace: Found {len(filenames)} images in {source_dir}')
    dataset = ImgDataset(filenames, trans)
    loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False)

    with torch.no_grad():
        for input_, paths in tqdm(loader, desc='QMagFace: Embedding'):
            input_ = input_.to(device)
            output = model(input_)
            if isinstance(output, (tuple, list)):
                embeddings = output[0].to('cpu')
            else:
                embeddings = output.to('cpu')
            embeddings = embeddings.numpy()
            for emb, img_path in zip(embeddings, paths):
                save_embedding_for_path(emb, img_path, source_dir, result_dir)

    print(f"QMagFace: Saved all embeddings to {result_dir} with mirrored folder structure.")

# ------------------ Main CLI ------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract face embeddings using ArcFace and QMagFace."
    )
    parser.add_argument('--input', type=str, default='data/lfw_sklearn',
                        help='Input image root directory')
    parser.add_argument('--arcface_output', type=str, default='results/arcface_embeddings_sklearn_v2',
                        help='ArcFace embeddings output directory')
    parser.add_argument('--qmagface_output', type=str, default='results/qmagface_embeddings_sklearn_v2',
                        help='QMagFace embeddings output directory')
    parser.add_argument('--qmagface_model', type=str, default='models/magface_model.pth',
                        help='Path to QMagFace model weights')
    parser.add_argument('--arcface', action='store_true',
                        help='Extract ArcFace embeddings')
    parser.add_argument('--qmagface', action='store_true',
                        help='Extract QMagFace embeddings')
    parser.add_argument('--both', action='store_true',
                        help='Extract both ArcFace and QMagFace embeddings')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for dataloaders')

    args = parser.parse_args()

    if args.arcface or args.both:
        extract_arcface_embeddings_deepface(args.input, args.arcface_output, batch_size=args.batch_size)
    if args.qmagface or args.both:
        extract_qmagface_embeddings(
            args.input, args.qmagface_output, args.qmagface_model, batch_size=args.batch_size
        )
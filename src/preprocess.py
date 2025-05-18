import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from insightface.app import FaceAnalysis
from facenet_pytorch import MTCNN
import argparse
import shutil
from pathlib import Path

class FacePreprocessor:
    def __init__(self, config):
        self.config = config
        
        
        # Initialize either MTCNN or InsightFace based on configuration
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and not config.cpu else 'cpu')
        
        if config.detector == 'mtcnn':
            self.detector = MTCNN(
                image_size=config.image_size,
                margin=config.margin,
                device=self.device,
                selection_method='probability',
                keep_all=False
            )
        elif config.detector == 'insightface':
            self.detector = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider' if str(self.device) != 'cpu' else 'CPUExecutionProvider']
            )
            self.detector.prepare(ctx_id=0 if str(self.device) != 'cpu' else -1, det_size=(640, 640))
        else:
            raise ValueError(f"Unknown detector: {config.detector}")
        
        print(f"Using device: {self.device}")
        print(f"Using detector: {config.detector}")
    
    
    def align_face_mtcnn(self, image):
        """Detect and align face using MTCNN, following ArcFace alignment procedure"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces and landmarks on full image (no pre-cropping)
        batch_boxes, batch_probs, batch_landmarks = self.detector.detect(img_rgb, landmarks=True)
        
        if batch_boxes is None or len(batch_boxes) == 0:
            return None
        
        # Take the face with highest probability if multiple faces detected
        if isinstance(batch_boxes, list):
            best_idx = 0
            if len(batch_probs) > 1:
                best_idx = np.argmax(batch_probs)
            box, landmarks = batch_boxes[best_idx], batch_landmarks[best_idx]
        else:
            box, landmarks = batch_boxes[0], batch_landmarks[0]
        
        # Get landmarks as numpy array (5 points: eyes, nose, mouth corners)
        landmarks = landmarks.astype(np.float32)
        
        # Define ArcFace reference template for 112x112 face alignment
        # These are the standard landmark positions used by ArcFace
        arcface_ref = np.array([
            [38.2946, 51.6963],  # left eye
            [73.5318, 51.5014],  # right eye
            [56.0252, 71.7366],  # nose
            [41.5493, 92.3655],  # left mouth corner
            [70.7299, 92.2041]   # right mouth corner
        ], dtype=np.float32)
        
        # Calculate similarity transform matrix
        transform_matrix = cv2.estimateAffinePartial2D(landmarks, arcface_ref)[0]
        
        # Apply affine transformation
        aligned_face = cv2.warpAffine(image, transform_matrix, (112, 112), 
                                      borderMode=cv2.BORDER_CONSTANT, 
                                      borderValue=(0, 0, 0))
            
        return aligned_face
    
    def align_face_insightface(self, image):
        """Detect and align face using InsightFace"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detector.get(img_rgb)
        
        if faces is None or len(faces) == 0:
            return None
        
        # Take the face with highest score if multiple faces detected
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
        
        face = faces[0]
        
        # InsightFace's built-in alignment also uses similar reference points to ArcFace
        # as both are from the same authors/framework
        aligned_face = face.face_chip
        
        # Convert back to BGR for OpenCV
        aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
        
        return aligned_face
    
    def process_dataset(self):
        """Process the entire LFW dataset"""
        lfw_dir = os.path.join(self.config.data_dir, self.config.input_dir)
        print(lfw_dir)
        aligned_dir = os.path.join(self.config.data_dir, self.config.output_dir)
        
        if not os.path.exists(lfw_dir):
            raise FileNotFoundError(f"Dataset directory not found at {lfw_dir}. Please extract the dataset first.")
        
        # Find identities with minimum number of images
        identities = {}
        person_dirs = os.listdir(lfw_dir)
        
        for person_dir in person_dirs:
            person_path = os.path.join(lfw_dir, person_dir)
            if os.path.isdir(person_path):
                image_files = [f for f in os.listdir(person_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(image_files) >= self.config.min_images:
                    identities[person_dir] = image_files
        
        print(f"Found {len(identities)} identities with at least {self.config.min_images} images")
        
        # Count total images for progress bar
        total_images = sum(len(files) for files in identities.values())
        print(f"Processing {total_images} images from {len(identities)} identities...")
        
        # Process each identity directory
        failed_images = 0
        with tqdm(total=total_images) as pbar:
            for person_dir, image_files in identities.items():
                person_path = os.path.join(lfw_dir, person_dir)
                
                # Create directory for aligned faces
                aligned_person_dir = os.path.join(aligned_dir, person_dir)
                os.makedirs(aligned_person_dir, exist_ok=True)
                
                # Process each image
                for img_name in sorted(image_files):
                    img_path = os.path.join(person_path, img_name)
                    aligned_img_path = os.path.join(aligned_person_dir, img_name)
                    
                    # Skip if already processed
                    if os.path.exists(aligned_img_path) and not self.config.force:
                        pbar.update(1)
                        continue
                    
                    # Read image
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Warning: Could not read image {img_path}")
                        failed_images += 1
                        pbar.update(1)
                        continue
                    
                    # Align face
                    if self.config.detector == 'mtcnn':
                        aligned_face = self.align_face_mtcnn(image)
                    else:
                        aligned_face = self.align_face_insightface(image)
                    
                    # Save aligned face if detection was successful
                    if aligned_face is not None:
                        cv2.imwrite(aligned_img_path, aligned_face)
                    else:
                        print(f"Warning: No face detected in {img_path}")
                        # Copy original image if specified
                        if self.config.keep_original_on_failure:
                            shutil.copy(img_path, aligned_img_path)
                        failed_images += 1
                    
                    pbar.update(1)
        
        print(f"Done! Processed {total_images} images with {failed_images} failures.")
        if failed_images > 0:
            print(f"Failure rate: {failed_images/total_images*100:.2f}%")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Face detection and alignment for LFW dataset')
    parser.add_argument('--detector', type=str, choices=['mtcnn', 'insightface'], default='mtcnn',
                        help='Face detector to use (MTCNN or InsightFace)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory containing the dataset')
    parser.add_argument('--input-dir', type=str, default='lfw/lfw-deepfunneled',
                        help='Input directory name within the data directory')
    parser.add_argument('--output-dir', type=str, default='aligned_lfw',
                        help='Output directory name within the data directory')
    parser.add_argument('--min-images', type=int, default=1,
                        help='Minimum number of images per person to process')
    parser.add_argument('--image-size', type=int, default=112, 
                        help='Size of aligned face images')
    parser.add_argument('--margin', type=int, default=0,
                        help='Margin around face for MTCNN')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing of already aligned images')
    parser.add_argument('--cpu', action='store_true',
                        help='Force using CPU even if GPU is available')
    parser.add_argument('--keep-original-on-failure', action='store_true',
                        help='Keep original image when face detection fails')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    preprocessor = FacePreprocessor(args)
    preprocessor.process_dataset() 
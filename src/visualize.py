import os
from utils import load_image, ensure_dir
from PIL import Image, ImageDraw, ImageFont

def save_misclassified_samples(
        misclassified_pairs, output_dir, 
        aligned_root='data/aligned_lfw', 
        y_true=None, y_pred=None, y_score=None, 
        max_examples=20):
    ensure_dir(output_dir)
    # Try to use a nice font if available
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
    for idx, (f1, f2, label) in enumerate(misclassified_pairs[:max_examples]):
        # Find aligned image paths
        parts1 = os.path.normpath(f1).split(os.sep)
        parts2 = os.path.normpath(f2).split(os.sep)
        identity1, filename1 = parts1[-2], parts1[-1].replace('.npy', '.jpg')
        identity2, filename2 = parts2[-2], parts2[-1].replace('.npy', '.jpg')
        img1_path = os.path.join(aligned_root, identity1, filename1)
        img2_path = os.path.join(aligned_root, identity2, filename2)
        if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
            print(f"Missing: {img1_path} or {img2_path}")
            continue

        im1 = load_image(img1_path).resize((112,112))
        im2 = load_image(img2_path).resize((112,112))
        # Create combined image with label bar (top) and border between
        combined = Image.new('RGB', (228, 146), (245,245,245))
        # Draw label bar
        draw = ImageDraw.Draw(combined)
        label_bar_height = 34
        draw.rectangle([0,0,228,label_bar_height], fill=(230,230,230))

        # Paste faces with a vertical margin
        combined.paste(im1, (6, label_bar_height+2))
        combined.paste(im2, (112+10, label_bar_height+2))

        # Draw border line
        draw.line((112+7, label_bar_height+2, 112+7, 146), fill=(160,160,160), width=2)

        # Prepare labels
        gt_label = "Genuine" if label == 1 else "Impostor"
        pred_label = "Genuine" if y_pred and y_pred[idx] == 1 else "Impostor"
        gt_color = (0,128,0) if label == 1 else (160,0,0)
        pred_color = (0,128,0) if y_pred and y_pred[idx] == 1 else (160,0,0)
        # Cosine score
        score_txt = f"Score: {y_score[idx]:.2f}" if y_score is not None else ""

        # Draw GT and Pred labels, color-coded
        draw.text((10, 6), f"GT: {gt_label}", fill=gt_color, font=font)
        draw.text((118, 6), f"Pred: {pred_label}", fill=pred_color, font=font)
        # Draw the cosine similarity in blue, centered
        # Fix: Pillow compatibility
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0,0), score_txt, font=font)
            w = bbox[2] - bbox[0]
        else:
            w, _ = font.getsize(score_txt)
        draw.text(((228-w)//2, label_bar_height-4), score_txt, fill=(28,28,190), font=font)
        combined.save(os.path.join(output_dir, f'misclassified_{idx}_true{label}_pred{y_pred[idx] if y_pred is not None else "?"}.png'))

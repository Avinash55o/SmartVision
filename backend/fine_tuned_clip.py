import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
import clip
from transformers import AdamW
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    logger.warning("CUDA not available, falling back to CPU.")

MEDICAL_LABELS = [
    "normal lung", "healthy chest", "normal bone", "healthy brain",
    "lung tumor", "pneumonia", "pulmonary nodule", "pneumothorax", "pleural effusion",
    "pulmonary edema", "rib fracture", "lung consolidation", "cardiomegaly",
    "brain tumor", "intracranial hemorrhage", "cerebral edema",
    "bone fracture", "bone tumor", "infection", "mass", "calcification", "edema"
]
BONE_CLASSES = ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 
                'humerus', 'shoulder fracture', 'wrist positive']

def load_data():
    data = []
    
    img_dir = "/app/datasets/chest_xray/images"
    csv_path = "/app/datasets/chest_xray/Data_Entry_2017.csv"
    if not os.path.exists(csv_path):
        logger.error(f"Chest X-ray CSV not found at {csv_path}")
    elif not os.path.exists(img_dir):
        logger.error(f"Chest X-ray images directory not found at {img_dir}")
    else:
        try:
            chest_df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(chest_df)} entries from Chest X-ray CSV")
            available_images = set(os.listdir(img_dir))
            for idx, row in chest_df.iterrows():
                img_name = row["Image Index"]
                if img_name in available_images:
                    img_path = os.path.join(img_dir, img_name)
                    labels = row["Finding Labels"].split("|")
                    for label in labels:
                        if label == "No Finding":
                            data.append((img_path, "normal lung"))
                        elif label in ["Mass", "Nodule"]:
                            data.append((img_path, "lung tumor" if label == "Mass" else "pulmonary nodule"))
                        elif label in ["Pneumonia", "Effusion", "Pneumothorax", "Edema", "Cardiomegaly", "Consolidation"]:
                            data.append((img_path, label.lower().replace("effusion", "pleural effusion")))
        except Exception as e:
            logger.error(f"Failed to load Chest X-ray CSV: {e}")

    bone_base = "/app/datasets/bone_fracture"
    if not os.path.exists(bone_base):
        logger.error(f"Bone Fracture directory not found at {bone_base}")
        return data

    split = "train"
    img_dir = os.path.join(bone_base, split, "images")
    label_dir = os.path.join(bone_base, split, "labels")
    if not (os.path.exists(img_dir) and os.path.exists(label_dir)):
        logger.warning(f"Skipping {split}: images or labels directory missing")
    else:
        for img_file in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_file)
            label_file = os.path.join(label_dir, img_file.replace(".jpg", ".txt"))
            if os.path.exists(label_file):
                try:
                    with open(label_file, "r") as f:
                        lines = f.readlines()
                        if lines:
                            for line in lines:
                                class_id = int(line.split()[0])
                                if class_id >= len(BONE_CLASSES):
                                    logger.warning(f"Invalid class_id {class_id} in {label_file}")
                                    continue
                                label = BONE_CLASSES[class_id]
                                if "fracture" in label or "positive" in label:
                                    data.append((img_path, "bone fracture"))
                                elif label == "humerus":
                                    data.append((img_path, "normal bone"))
                except Exception as e:
                    logger.error(f"Error reading {label_file}: {e}")
            else:
                data.append((img_path, "normal bone"))

    logger.info(f"Loaded {len(data)} total image-label pairs")
    return data

try:
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.train()
except RuntimeError as e:
    if "checksum does not match" in str(e):
        logger.error(f"Checksum mismatch: {e}. Clearing cache and retrying.")
        import shutil
        shutil.rmtree(os.path.expanduser("~/.cache/clip"), ignore_errors=True)
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.train()
    else:
        logger.error(f"Failed to load CLIP model: {e}")
        exit(1)

optimizer = AdamW(model.parameters(), lr=5e-5)

data = load_data()
if not data:
    logger.error("No data loaded. Check dataset paths and contents.")
    exit(1)

logger.info("Starting training with chunked data")
chunk_size = 1000
for epoch in range(2):  # Changed to 2 epochs
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        logger.info(f"Epoch {epoch}, Processing chunk {i // chunk_size + 1} of {len(data) // chunk_size + 1}")
        
        dataset = []
        for img_path, label in chunk:
            try:
                img = preprocess(Image.open(img_path))
                text = clip.tokenize(label).squeeze(0)
                dataset.append((img, text))
            except Exception as e:
                logger.warning(f"Skipping {img_path}: {e}")
        
        if not dataset:
            logger.warning(f"Chunk {i // chunk_size + 1} is empty, skipping")
            continue
        
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
        for batch_idx, (images, texts) in enumerate(dataloader):
            try:
                logger.debug(f"Before fix - images: {images.shape}, texts: {texts.shape}")
                images = images.to(device)
                texts = texts.to(device)
                if texts.dim() == 3:
                    texts = texts.squeeze(1)
                elif texts.dim() == 1:
                    texts = texts.unsqueeze(0)
                logger.debug(f"After fix - images: {images.shape}, texts: {texts.shape}")
                optimizer.zero_grad()
                logits_per_image, _ = model(images, texts)
                labels = torch.arange(len(images)).to(device)
                loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch}, Chunk {i // chunk_size + 1}, Batch {batch_idx}, Loss: {loss.item()}")
            except Exception as e:
                logger.error(f"Training error in epoch {epoch}, chunk {i // chunk_size + 1}: {e}")
        dataset = []
    logger.info(f"Epoch {epoch} completed")

output_path = "/app/fine_tuned_clip.pt"
try:
    torch.save(model.state_dict(), output_path)
    logger.info(f"Model saved to {output_path}")
except Exception as e:
    logger.error(f"Failed to save model: {e}")
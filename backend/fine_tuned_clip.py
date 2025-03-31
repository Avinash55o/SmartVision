import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
import clip
from transformers import AdamW
import pandas as pd
import logging
import sys

print("Starting fine_tuned_clip.py script...")

# Set up file logging
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fine_tuned_clip.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

with open(log_file, "w") as f:
    f.write("Log file initialized\n")

print(f"Logs will be written to: {log_file}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
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
    print("Starting to load data...")
    data = []
    
    # Get base directory (current directory or default to Docker paths)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Base directory: {base_dir}")
    
    img_dir = os.path.join(base_dir, "datasets", "chest_xray", "images")
    csv_path = os.path.join(base_dir, "datasets", "chest_xray", "Data_Entry_2017.csv")
    
    # Try Docker paths as fallback
    if not os.path.exists(csv_path):
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

    # Try local path first, then Docker path
    bone_base = os.path.join(base_dir, "datasets", "bone_fracture")
    if not os.path.exists(bone_base):
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

print("About to try loading the CLIP model...")
try:
    # Try loading the base CLIP model
    print("Calling clip.load...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.train()
    print("CLIP model loaded successfully!")
except RuntimeError as e:
    if "checksum does not match" in str(e):
        logger.error(f"Checksum mismatch: {e}. Clearing cache and retrying.")
        import shutil
        shutil.rmtree(os.path.expanduser("~/.cache/clip"), ignore_errors=True)
        try:
            model, preprocess = clip.load("ViT-B/32", device=device)
            model.train()
        except Exception as retry_e:
            logger.error(f"Failed to load CLIP model after cache clear: {retry_e}")
            exit(1)
    else:
        logger.error(f"Failed to load CLIP model: {e}")
        exit(1)
except Exception as e:
    logger.error(f"Unexpected error loading CLIP model: {e}")
    exit(1)

logger.info(f"Successfully loaded CLIP model on {device}")

# Training hyperparameters
BATCH_SIZE = 32
CHUNK_SIZE = 500  # Reduced from 1000 for better memory management
LEARNING_RATE = 1e-5  # Reduced from 5e-5 for more stable training
NUM_EPOCHS = 5  # Increased from 2 for better convergence
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.01

# Initialize optimizer with better parameters
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
    eps=1e-8
)

# First load the data
data = load_data()
if not data:
    logger.error("No data loaded. Check dataset paths and contents.")
    exit(1)

# Create learning rate scheduler after data is loaded
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    epochs=NUM_EPOCHS,
    steps_per_epoch=len(data) // BATCH_SIZE,
    pct_start=0.1
)

logger.info(f"Starting training with {len(data)} samples")
logger.info(f"Training parameters: batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}, lr={LEARNING_RATE}")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for i in range(0, len(data), CHUNK_SIZE):
        chunk = data[i:i + CHUNK_SIZE]
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Processing chunk {i // CHUNK_SIZE + 1} of {len(data) // CHUNK_SIZE + 1}")
        
        dataset = []
        for img_path, label in chunk:
            try:
                img = preprocess(Image.open(img_path))
                text = clip.tokenize(label).squeeze(0)
                dataset.append((img, text))
            except Exception as e:
                logger.warning(f"Skipping {img_path}: {e}")
        
        if not dataset:
            logger.warning(f"Chunk {i // CHUNK_SIZE + 1} is empty, skipping")
            continue
        
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        
        for batch_idx, (images, texts) in enumerate(dataloader):
            try:
                images = images.to(device)
                texts = texts.to(device)
                
                if texts.dim() == 3:
                    texts = texts.squeeze(1)
                elif texts.dim() == 1:
                    texts = texts.unsqueeze(0)
                
                optimizer.zero_grad()
                logits_per_image, _ = model(images, texts)
                labels = torch.arange(len(images)).to(device)
                loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    avg_loss = total_loss / num_batches
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(f"Epoch {epoch + 1}, Chunk {i // CHUNK_SIZE + 1}, Batch {batch_idx}, "
                              f"Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
                    
            except Exception as e:
                logger.error(f"Training error in epoch {epoch + 1}, chunk {i // CHUNK_SIZE + 1}: {e}")
                continue
        
        # Clear memory after each chunk
        torch.cuda.empty_cache()
        dataset = []
    
    avg_epoch_loss = total_loss / num_batches
    logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

# Save the model with additional metadata
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fine_tuned_clip.pt")
try:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': NUM_EPOCHS,
        'loss': avg_epoch_loss,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'chunk_size': CHUNK_SIZE
        }
    }, output_path)
    logger.info(f"Model and training metadata saved to {output_path}")
except Exception as e:
    logger.error(f"Failed to save model: {e}")
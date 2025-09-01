import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DPTForDepthEstimation, DPTImageProcessor
from PIL import Image
import os
import argparse
from tqdm import tqdm
import numpy as np

from common.config.parser import load_config

def get_best_device(requested_device: str = "auto") -> torch.device:
    """
    Detects and returns the best available computing device as a torch.device object.
    """
    if requested_device == "auto":
        if torch.cuda.is_available():
            print("CUDA disponível. Usando GPU.")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Apple MPS disponível. Usando GPU.")
            return torch.device("mps")
        else:
            print("Nenhuma GPU detectada. Usando CPU.")
            return torch.device("cpu")
    else:
        print(f"Dispositivo solicitado: {requested_device}")
        return torch.device(requested_device)

class CustomDepthDataset(Dataset):
    """
    Custom PyTorch Dataset for loading RGB images and their corresponding depth maps.
    """
    def __init__(self, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.rgb_path = os.path.join(self.root_dir, 'rgb')
        self.depth_path = os.path.join(self.root_dir, 'depth')
        self.image_files = sorted([f for f in os.listdir(self.rgb_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        rgb_image_path = os.path.join(self.rgb_path, image_name)
        depth_image_name = os.path.splitext(image_name)[0] + '.png'
        depth_image_path = os.path.join(self.depth_path, depth_image_name)

        image = Image.open(rgb_image_path).convert("RGB")
        depth_map = np.array(Image.open(depth_image_path))

        inputs = self.processor(images=image, labels=depth_map, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].squeeze(0)
        inputs["labels"] = inputs["labels"].squeeze(0)
        return inputs

def train_one_epoch(model, loader, optimizer, device):
    """Performs one training epoch."""
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate_one_epoch(model, loader, device):
    """Performs one validation epoch."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(loader)

def main(config_path: str):
    """Main function to run the fine-tuning pipeline."""
    # 1. Load Configuration
    config = load_config(config_path)

    # 2. Setup Device
    device = get_best_device(config.hardware.device)

    # 3. Load Model and Processor
    print(f"Carregando modelo pré-treinado: {config.model_name}")
    model = DPTForDepthEstimation.from_pretrained(config.model_name)
    processor = DPTImageProcessor.from_pretrained(config.model_name)
    model.to(device)

    # 4. Create Datasets and DataLoaders
    print("Criando datasets e dataloaders...")
    train_dataset = CustomDepthDataset(root_dir=config.dataset.train_dir, processor=processor)
    val_dataset = CustomDepthDataset(root_dir=config.dataset.val_dir, processor=processor)
    train_loader = DataLoader(train_dataset, batch_size=config.training_hyperparameters.batch_size, shuffle=True, num_workers=config.hardware.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.training_hyperparameters.batch_size, shuffle=False, num_workers=config.hardware.num_workers)

    # 5. Setup Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.training_hyperparameters.learning_rate)

    # 6. Training and Validation Loop
    best_val_loss = float('inf')
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*40)
    print("Iniciando Fine-Tuning do MiDaS (DPT)")
    print(f"   - Épocas: {config.training_hyperparameters.epochs}")
    print(f"   - Batch Size: {config.training_hyperparameters.batch_size}")
    print(f"   - Learning Rate: {config.training_hyperparameters.learning_rate}")
    print(f"   - Dispositivo: {device.type.upper()}")
    print("="*40 + "\n")

    for epoch in range(config.training_hyperparameters.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch+1}/{config.training_hyperparameters.epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            print(f"Novo melhor modelo salvo em: {output_dir} (Val Loss: {best_val_loss:.4f})")

    print("\n" + "="*40)
    print("Fine-tuning concluído!")
    print(f"O melhor modelo foi salvo em: {output_dir}")
    print("="*40 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de Fine-Tuning para MiDaS (DPT) com Configuração YAML")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/midas_finetune.yml',
        help='Caminho para o arquivo de configuração YAML.'
    )
    args = parser.parse_args()
    main(args.config)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DPTForDepthEstimation, DPTImageProcessor
from PIL import Image
import os
import argparse
from tqdm import tqdm
import numpy as np

def get_best_device():
    """Detects and returns the best available computing device."""
    if torch.cuda.is_available():
        print("CUDA disponível. Usando GPU.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Apple MPS disponível. Usando GPU.")
        return torch.device("mps")
    else:
        print("Nenhuma GPU detectada. Usando CPU.")
        return torch.device("cpu")

class CustomDepthDataset(Dataset):
    """
    Custom PyTorch Dataset for loading RGB images and their corresponding depth maps.
    """
    def __init__(self, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.rgb_path = os.path.join(self.root_dir, 'rgb')
        self.depth_path = os.path.join(self.root_dir, 'depth')

        # Assumes filenames in rgb and depth folders match
        self.image_files = sorted([f for f in os.listdir(self.rgb_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]

        rgb_image_path = os.path.join(self.rgb_path, image_name)
        # Assumes depth image has same name but could be different format (e.g., .png)
        depth_image_name = os.path.splitext(image_name)[0] + '.png'
        depth_image_path = os.path.join(self.depth_path, depth_image_name)

        image = Image.open(rgb_image_path).convert("RGB")

        # Load depth map as a numpy array.
        # Using PIL to open and convert to a NumPy array.
        # Assumes depth is stored in a format readable by PIL (e.g., 16-bit PNG).
        depth_map = Image.open(depth_image_path)
        depth_map = np.array(depth_map)

        # The DPT model expects the labels (depth maps) to be pre-processed.
        # The processor handles both image and depth map preparation.
        # It will resize, normalize, and format both the image and the depth map.
        # The 'labels' are the processed depth maps.
        inputs = self.processor(images=image, labels=depth_map, return_tensors="pt")

        # Squeeze to remove the batch dimension added by the processor
        inputs["pixel_values"] = inputs["pixel_values"].squeeze(0)
        inputs["labels"] = inputs["labels"].squeeze(0)

        return inputs

def train_one_epoch(model, loader, optimizer, device):
    """Performs one training epoch."""
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training"):
        # Move batch to device
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass - model computes loss internally when labels are provided
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

def main(args):
    """Main function to run the fine-tuning pipeline."""
    device = get_best_device()

    # --- 1. Load Model and Processor ---
    print(f"Carregando modelo pré-treinado: {args.model_name}")
    model = DPTForDepthEstimation.from_pretrained(args.model_name)
    processor = DPTImageProcessor.from_pretrained(args.model_name)
    model.to(device)

    # --- 2. Create Datasets and DataLoaders ---
    print("Criando datasets e dataloaders...")
    train_dataset = CustomDepthDataset(root_dir=args.train_dir, processor=processor)
    val_dataset = CustomDepthDataset(root_dir=args.val_dir, processor=processor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- 3. Setup Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # --- 4. Training and Validation Loop ---
    best_val_loss = float('inf')
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*40)
    print("Iniciando Fine-Tuning do MiDaS (DPT)")
    print(f"   - Épocas: {args.epochs}")
    print(f"   - Batch Size: {args.batch_size}")
    print(f"   - Learning Rate: {args.learning_rate}")
    print(f"   - Dispositivo: {device.type.upper()}")
    print("="*40 + "\n")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the processor along with the model for easy reloading
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            print(f"Novo melhor modelo salvo em: {output_dir} (Val Loss: {best_val_loss:.4f})")

    print("\n" + "="*40)
    print("Fine-tuning concluído!")
    print(f"O melhor modelo foi salvo em: {output_dir}")
    print("="*40 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de Fine-Tuning para MiDaS (DPT)")

    parser.add_argument('--model-name', type=str, default="Intel/dpt-swinv2-tiny-256", help='Nome do modelo DPT do Hugging Face.')
    parser.add_argument('--train-dir', type=str, default='./datasets/midas_custom/train', help='Diretório com os dados de treinamento.')
    parser.add_argument('--val-dir', type=str, default='./datasets/midas_custom/val', help='Diretório com os dados de validação.')
    parser.add_argument('--output-dir', type=str, default='./models/midas_finetuned', help='Diretório para salvar o modelo treinado.')

    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas de treinamento.')
    parser.add_argument('--batch-size', type=int, default=4, help='Tamanho do lote (batch size). Reduza se tiver erros de memória (OOM).')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Taxa de aprendizado.')
    parser.add_argument('--num-workers', type=int, default=2, help='Número de workers para o DataLoader.')

    args = parser.parse_args()
    main(args)

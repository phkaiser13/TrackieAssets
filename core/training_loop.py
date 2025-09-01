import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import logging

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_one_epoch(
    model: Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """
    Executa uma única época de treinamento para o modelo fornecido.

    Args:
        model (Module): O modelo a ser treinado.
        data_loader (DataLoader): O DataLoader que fornece os dados de treinamento.
        optimizer (Optimizer): O otimizador para atualização dos pesos.
        device (torch.device): O dispositivo (CPU ou GPU) onde o treinamento será executado.
        epoch (int): O número da época atual, usado para logging.

    Returns:
        float: A perda (loss) média da época.
    """
    model.train()  # Coloca o modelo em modo de treinamento
    total_loss = 0.0

    # Adiciona uma barra de progresso para visualização
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}", leave=True)

    for images, targets in progress_bar:
        # Mover dados para o dispositivo de destino (GPU/CPU)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 1. Zerar os gradientes do otimizador
        optimizer.zero_grad()

        # 2. Forward pass: obter as predições do modelo
        predictions = model(images)

        # 3. Calcular a perda
        # A lógica de cálculo de perda é delegada ao próprio modelo
        loss = model.get_loss(predictions, targets)

        # 4. Backward pass: calcular os gradientes da perda
        loss.backward()

        # 5. Atualizar os pesos do modelo
        optimizer.step()

        # Acumular a perda para calcular a média no final
        total_loss += loss.item()

        # Atualizar a barra de progresso com a perda atual
        progress_bar.set_postfix(loss=loss.item())

    # Calcular a perda média da época
    avg_loss = total_loss / len(data_loader)
    logging.info(f"Fim da Epoch {epoch+1} - Perda Média: {avg_loss:.4f}")

    return avg_loss

# Este arquivo é um módulo e não se destina a ser executado diretamente.
# A lógica de orquestração estará em `train_yolo.py`.

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Optional

# Em um cenário real, isso viria de um arquivo de dataset real.
# Por enquanto, é um placeholder para a estrutura.
class DummyDataset(Dataset):
    """
    Um dataset de exemplo para ilustrar a estrutura do data loader.
    Ele gera tensores aleatórios para imagens e labels.
    """
    def __init__(self, num_samples=1000, img_size=640, num_classes=80):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Gera uma imagem e um alvo falsos
        image = torch.randn(3, self.img_size, self.img_size)

        # Gera um alvo falso no formato [class, x_center, y_center, width, height]
        target = torch.rand(1, 5)
        target[:, 0] = torch.randint(0, self.num_classes, (1,))

        return image, target

def create_data_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    collate_fn: Optional[Callable] = None
) -> DataLoader:
    """
    Cria e retorna um DataLoader do PyTorch.

    Args:
        dataset (Dataset): O dataset a ser carregado.
        batch_size (int): O tamanho do batch.
        num_workers (int): O número de subprocessos para carregamento de dados.
        shuffle (bool): Se os dados devem ser embaralhados a cada época.
        collate_fn (Optional[Callable]): Uma função opcional para mesclar uma lista de amostras em um batch.

    Returns:
        DataLoader: A instância do DataLoader configurada.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True, # Otimização para transferência de dados para a GPU
        persistent_workers=True if num_workers > 0 else False
    )

# Exemplo de como usar (será chamado pelo script de treinamento principal)
if __name__ == '__main__':
    # Configuração de exemplo
    BATCH_SIZE = 16
    IMG_SIZE = 640
    NUM_WORKERS = 4

    # 1. Criar o dataset
    # Em um caso real, você passaria o caminho do seu dataset aqui
    train_dataset = DummyDataset(img_size=IMG_SIZE)

    # 2. Criar o data loader
    # O collate_fn será customizado no futuro para incluir augmentations da GPU
    train_loader = create_data_loader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True
    )

    # 3. Iterar sobre o data loader para verificar
    print(f"DataLoader criado com {len(train_loader)} batches de tamanho {BATCH_SIZE}.")
    images, labels = next(iter(train_loader))
    print(f"Shape do batch de imagens: {images.shape}") # Esperado: [BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE]
    print(f"Shape do batch de labels: {labels.shape}") # Esperado: [BATCH_SIZE, 1, 5]

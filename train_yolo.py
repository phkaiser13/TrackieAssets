import torch
import argparse
import os
import sys
import platform
import ctypes
import numpy as np
import logging

# Adiciona o diretório raiz ao path para permitir importações de `core`, `models`, etc.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common.config.parser import load_config
from core.data_loader import create_data_loader, DummyDataset
from core.training_loop import train_one_epoch
from models.yolov5nu.model import YOLOv5nu

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Carregador do Backend Acelerado ---
def load_backend_lib(config):
    """
    Detecta o SO, encontra e carrega a biblioteca do backend compilada (.so ou .dylib).
    Os engenheiros do usuário são responsáveis por compilar os backends.
    Este script espera que a biblioteca esteja em `backends/build/`.
    """
    backend_dir = os.path.join('backends', 'build')
    system = platform.system()

    if system == 'Linux' and config.hardware.device == 'cuda':
        lib_name = 'libcuda_augment.so'
        func_name = 'apply_brightness_cuda'
    elif system == 'Darwin' and config.hardware.device == 'mps': # 'mps' é usado como análogo para Metal
        lib_name = 'libmetal_augment.dylib'
        func_name = 'apply_brightness_metal'
    else:
        logging.warning(f"Nenhum backend acelerado para o sistema '{system}' e dispositivo '{config.hardware.device}'. A augmentation da GPU será desativada.")
        return None, None

    lib_path = os.path.join(backend_dir, lib_name)
    if not os.path.exists(lib_path):
        logging.error(f"Biblioteca do backend não encontrada em '{lib_path}'.")
        logging.error("Certifique-se de que os backends foram compilados (ex: executando 'make' em `backends/`).")
        logging.error("A augmentation da GPU será desativada.")
        return None, None

    try:
        lib = ctypes.CDLL(lib_path)
        func = getattr(lib, func_name)

        # Define os tipos de argumentos para a função C
        func.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float
        ]
        func.restype = None
        logging.info(f"Biblioteca do backend '{lib_name}' carregada com sucesso.")
        return func, config.hardware.device
    except (OSError, AttributeError) as e:
        logging.error(f"Falha ao carregar o backend de '{lib_path}': {e}")
        return None, None

# --- Collate Function Customizada com Augmentation ---
def create_augmenting_collate_fn(backend_func, brightness_factor):
    """Cria uma collate_fn que aplica a augmentation da GPU a um batch de imagens."""
    if not backend_func:
        return None

    def augment_collate(batch):
        images, labels = zip(*batch)
        images = torch.stack(images, 0)
        labels = torch.cat(labels, 0)

        # Prepara os dados para a chamada da função C
        # A função C espera um array numpy contíguo
        images_np = images.numpy().astype(np.float32, copy=False)
        output_np = np.empty_like(images_np)

        b, c, h, w = images_np.shape

        # Aplica a augmentation em cada imagem do batch
        for i in range(b):
            img_in = np.ascontiguousarray(images_np[i])
            img_out = np.ascontiguousarray(output_np[i])
            backend_func(img_in, img_out, w, h, c, brightness_factor)

        return torch.from_numpy(output_np), labels

    return augment_collate

# --- Função Principal de Treinamento ---
def main(config_path: str):
    """
    Função principal para orquestrar o pipeline de treinamento customizado.
    """
    config = load_config(config_path)

    # 1. Carregar o backend de hardware acelerado
    augment_func, device_type = load_backend_lib(config)

    # 2. Configurar o dispositivo de treinamento
    if device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device_type == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logging.info(f"Treinamento será executado no dispositivo: {device}")

    # 3. Criar o modelo
    model = YOLOv5nu(num_classes=config.dataset.num_classes).to(device)

    # 4. Preparar o dataset e o data loader
    dataset = DummyDataset(img_size=config.training_hyperparameters.img_size)

    collate_fn = None
    if augment_func and config.augmentation.augment:
        logging.info("Augmentation da GPU ativada.")
        collate_fn = create_augmenting_collate_fn(augment_func, config.augmentation.brightness)
    else:
        logging.info("Usando data loader padrão sem augmentation da GPU.")

    data_loader = create_data_loader(
        dataset,
        batch_size=config.training_hyperparameters.batch_size,
        num_workers=config.hardware.workers,
        collate_fn=collate_fn
    )

    # 5. Obter o otimizador
    optimizer_config = {
        "optimizer": config.training_hyperparameters.optimizer,
        "lr": config.training_hyperparameters.lr0
    }
    optimizer = model.get_optimizer(optimizer_config)

    # 6. Iniciar o loop de treinamento
    logging.info("="*50)
    logging.info("Iniciando Treinamento Customizado do Modelo YOLO")
    logging.info(f"   - Modelo: {model.__class__.__name__}")
    logging.info(f"   - Épocas: {config.training_hyperparameters.epochs}")
    logging.info(f"   - Dispositivo: {device.type.upper()}")
    logging.info("="*50)

    for epoch in range(config.training_hyperparameters.epochs):
        train_one_epoch(model, data_loader, optimizer, device, epoch)

    logging.info("Treinamento concluído com sucesso!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de Treinamento YOLO Customizado com Backends Acelerados")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/yolo_train.yml',
        help='Caminho para o arquivo de configuração YAML.'
    )
    args = parser.parse_args()
    main(args.config)

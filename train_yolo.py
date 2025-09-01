import torch
import argparse
import os
from ultralytics import YOLO

from common.config.parser import load_config

def get_best_device(requested_device: str = "auto") -> str:
    """
    Detects and returns the best available computing device based on user request.
    Prioritizes CUDA > MPS > CPU if 'auto' is selected.
    """
    if requested_device == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
            gpu_count = torch.cuda.device_count()
            print(f"CUDA disponível. Usando {gpu_count} GPU(s).")
        elif torch.backends.mps.is_available():
            device_name = "mps"
            print("Apple MPS (Metal Performance Shaders) disponível. Usando MPS.")
        else:
            device_name = "cpu"
            print("CUDA e MPS não encontrados. Usando CPU (treinamento será lento).")
    else:
        device_name = requested_device
        print(f"Dispositivo solicitado: {device_name}")

    if "cuda" in device_name:
        torch.cuda.empty_cache()

    return device_name

def main(config_path: str):
    """
    Função principal para executar o pipeline de treinamento do YOLO.
    """
    # 1. Load Configuration
    config = load_config(config_path)

    # 2. Obter o melhor dispositivo de hardware
    device = get_best_device(config.hardware.device)

    # 3. Verificar se o arquivo de dados YAML existe
    if not os.path.exists(config.dataset.data_yaml):
        raise FileNotFoundError(f"Arquivo de configuração de dados não encontrado em: {config.dataset.data_yaml}")

    # 4. Carregar o modelo YOLO.
    print(f"Carregando modelo pré-treinado: {config.model_name}")
    model = YOLO(config.model_name)
    model.to(device)

    # 5. Iniciar o treinamento
    print("\n" + "="*40)
    print("Iniciando o Treinamento do Modelo YOLO")
    print(f"   - Modelo: {config.model_name}")
    print(f"   - Dataset: {config.dataset.data_yaml}")
    print(f"   - Épocas: {config.training_hyperparameters.epochs}")
    print(f"   - Batch Size: {config.training_hyperparameters.batch_size}")
    print(f"   - Tamanho da Imagem: {config.training_hyperparameters.img_size}")
    print(f"   - Dispositivo: {device.upper()}")
    print("="*40 + "\n")

    results = model.train(
        # Dataset
        data=config.dataset.data_yaml,

        # Hyperparameters
        epochs=config.training_hyperparameters.epochs,
        batch=config.training_hyperparameters.batch_size,
        imgsz=config.training_hyperparameters.img_size,
        patience=config.training_hyperparameters.patience,
        optimizer=config.training_hyperparameters.optimizer,
        lr0=config.training_hyperparameters.lr0,
        lrf=config.training_hyperparameters.lrf,

        # Augmentation
        augment=config.augmentation.augment,
        mixup=config.augmentation.mixup,
        copy_paste=config.augmentation.copy_paste,
        dropout=config.augmentation.dropout,

        # Hardware & Project
        device=device,
        workers=config.hardware.workers,
        project=config.project_name,
        name='train_run',
        exist_ok=True,
    )

    print("\n" + "="*40)
    print("Treinamento concluído com sucesso!")
    print(f"Resultados, logs e pesos do modelo salvos em: {results.save_dir}")
    print(f"O melhor modelo foi salvo em: {os.path.join(results.save_dir, 'weights/best.pt')}")
    print("="*40 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de Treinamento YOLOv5/v8 com Configuração YAML")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/yolo_train.yml',
        help='Caminho para o arquivo de configuração YAML.'
    )
    args = parser.parse_args()
    main(args.config)

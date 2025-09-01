import torch
import argparse
import os
import yaml
from ultralytics import YOLO

def get_best_device():
    """
    Detecta e retorna o melhor dispositivo de computação disponível (CUDA, MPS ou CPU).
    Prioriza CUDA > MPS > CPU.
    """
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.cuda.empty_cache()
        gpu_count = torch.cuda.device_count()
        print(f"CUDA disponível. Usando {gpu_count} GPU(s).")
        # Para treinamento multi-GPU, o Ultralytics espera o ID do dispositivo, ex: [0, 1]
        # ou 0 para uma única GPU. Passar 'cuda' geralmente funciona para uma única GPU.
        # Para múltiplas, o ideal é rodar via CLI com 'yolo train ... device=0,1'
        # ou ajustar aqui para retornar uma lista de devices. Por simplicidade, retornamos a string.
    elif torch.backends.mps.is_available():
        device_name = "mps"
        print("Apple MPS (Metal Performance Shaders) disponível. Usando MPS.")
    else:
        device_name = "cpu"
        print("CUDA e MPS não encontrados. Usando CPU (treinamento será lento).")

    return device_name

def main(args):
    """
    Função principal para executar o pipeline de treinamento do YOLO.
    """
    # 1. Obter o melhor dispositivo de hardware
    device = get_best_device()

    # 2. Verificar se o arquivo de dados YAML existe
    if not os.path.exists(args.data_yaml):
        raise FileNotFoundError(f"Arquivo de configuração de dados não encontrado em: {args.data_yaml}")

    # 3. Carregar o modelo YOLO.
    # Carrega a arquitetura e os pesos pré-treinados (transfer learning).
    print(f"Carregando modelo pré-treinado: {args.model_name}")
    model = YOLO(args.model_name)
    model.to(device) # Mover o modelo para o dispositivo antes do treino

    # 4. Iniciar o treinamento
    print("\n" + "="*40)
    print("Iniciando o Treinamento do Modelo YOLO")
    print(f"   - Modelo: {args.model_name}")
    print(f"   - Dataset: {args.data_yaml}")
    print(f"   - Épocas: {args.epochs}")
    print(f"   - Batch Size: {args.batch_size}")
    print(f"   - Tamanho da Imagem: {args.img_size}")
    print(f"   - Dispositivo: {device.upper()}")
    print("="*40 + "\n")

    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=device,
        project=args.project_name,
        name='train_run', # O subdiretório específico para este treino
        exist_ok=True, # Permite sobrescrever treinos anteriores no mesmo dir
        patience=20 # Número de épocas sem melhora para parar o treino (Early Stopping)
    )

    print("\n" + "="*40)
    print("Treinamento concluído com sucesso!")
    # A biblioteca Ultralytics salva os resultados automaticamente.
    # O melhor modelo é salvo como 'best.pt' no diretório de resultados.
    # O diretório de resultados é algo como: 'runs/train/expN/weights/best.pt'
    print(f"Resultados, logs e pesos do modelo salvos em: {results.save_dir}")
    print(f"O melhor modelo foi salvo em: {os.path.join(results.save_dir, 'weights/best.pt')}")
    print("="*40 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de Treinamento YOLOv5/v8 com Ultralytics")

    parser.add_argument(
        '--data-yaml',
        type=str,
        default='yolo_custom_data.yaml',
        help='Caminho para o arquivo de configuração do dataset (.yaml).'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='yolov5nu.pt',
        help='Nome do modelo YOLO a ser treinado (ex: yolov5nu.pt, yolov8n.pt).'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Número total de épocas de treinamento.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Tamanho do lote (batch size) para o treinamento. Ajuste conforme a VRAM da sua GPU.'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Tamanho das imagens de entrada (altura e largura).'
    )
    parser.add_argument(
        '--project-name',
        type=str,
        default='runs/train',
        help='Nome do diretório do projeto onde os resultados serão salvos.'
    )

    args = parser.parse_args()
    main(args)

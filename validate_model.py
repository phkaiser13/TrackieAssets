import argparse
import os
from ultralytics import YOLO

def main(args):
    """
    Função principal para validar um modelo YOLO treinado em um conjunto de dados.
    """
    # 1. Verificar se os arquivos necessários existem
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Arquivo de pesos não encontrado em: {args.weights}")
    if not os.path.exists(args.data_yaml):
        raise FileNotFoundError(f"Arquivo de configuração de dados não encontrado em: {args.data_yaml}")

    # 2. Carregar o modelo treinado
    print(f"Carregando modelo de: {args.weights}")
    model = YOLO(args.weights)

    # 3. Executar a validação
    print("\n" + "="*40)
    print("Iniciando a Validação do Modelo")
    print(f"   - Modelo: {args.weights}")
    print(f"   - Dataset: {args.data_yaml}")
    print(f"   - Conjunto de Dados (Split): {args.split}")
    print(f"   - Tamanho da Imagem: {args.img_size}")
    print("="*40 + "\n")

    try:
        # O método `val` retorna um objeto de métricas com os resultados
        metrics = model.val(
            data=args.data_yaml,
            imgsz=args.img_size,
            split=args.split,
            project='runs/validation',
            name=f'val_run_{Path(args.weights).stem}'
        )

        print("Validação concluída com sucesso!")
        print("\n--- Resultados da Validação ---")

        # O objeto de métricas tem um dicionário `box` com as principais métricas de detecção
        box_metrics = metrics.box
        print(f"  mAP50-95: {box_metrics.map:.4f}")   # Mean Average Precision @ IoU=0.5:0.95
        print(f"  mAP50:    {box_metrics.map50:.4f}") # Mean Average Precision @ IoU=0.5
        print(f"  mAP75:    {box_metrics.map75:.4f}") # Mean Average Precision @ IoU=0.75
        print(f"  Precisão (Precision): {box_metrics.p[0]:.4f}") # Exemplo para a primeira classe
        print(f"  Revocação (Recall):   {box_metrics.r[0]:.4f}") # Exemplo para a primeira classe

        # Para ver todas as métricas detalhadas por classe:
        # metrics.box.print()
        # Ou acesse o pandas DataFrame:
        # print(metrics.box.maps) # mAP por classe

        print("\nResultados, gráficos e predições salvos em: runs/validation/")

    except Exception as e:
        print(f"Ocorreu um erro durante a validação: {e}")

    print("\n" + "="*40)

if __name__ == '__main__':
    # Importar Path aqui para ser usado no default do 'name'
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Script de Validação de Modelos YOLO")

    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help="Caminho para o arquivo de pesos do modelo treinado (ex: 'runs/train/train_run/weights/best.pt')."
    )
    parser.add_argument(
        '--data-yaml',
        type=str,
        default='yolo_custom_data.yaml',
        help='Caminho para o arquivo de configuração do dataset (.yaml).'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Tamanho das imagens de entrada (altura e largura) para a validação.'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['val', 'test'],
        help="Conjunto de dados a ser usado para a validação: 'val' ou 'test'."
    )

    args = parser.parse_args()
    main(args)

import argparse
import os
from pathlib import Path
from ultralytics import YOLO

def main(args):
    """
    Função principal para exportar um modelo YOLO treinado para o formato ONNX.
    """
    # 1. Verificar se o arquivo de pesos do modelo existe
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Arquivo de pesos não encontrado em: {args.weights}")

    print(f"Carregando modelo a partir de: {args.weights}")

    # 2. Carregar o modelo treinado
    # A classe YOLO infere o tipo de modelo a partir da extensão .pt
    model = YOLO(args.weights)

    # 3. Executar a exportação para ONNX
    print("\n" + "="*40)
    print("Iniciando a Exportação para ONNX")
    print(f"   - Modelo de Entrada: {args.weights}")
    print(f"   - Formato de Saída: ONNX")
    print(f"   - Tamanho da Imagem: {args.img_size}")
    print(f"   - Simplificar Modelo: {'Sim' if args.simplify else 'Não'}")
    print("="*40 + "\n")

    try:
        # A função export retorna o caminho para o arquivo exportado
        output_path = model.export(
            format='onnx',
            imgsz=args.img_size,
            simplify=args.simplify,
            opset=args.opset # Definir o opset pode ser importante para compatibilidade
        )
        print("Exportação para ONNX concluída com sucesso!")
        print(f"Modelo salvo em: {output_path}")

    except Exception as e:
        print(f"Ocorreu um erro durante a exportação: {e}")
        print("Verifique se as dependências 'onnx' e 'onnx-simplifier' estão instaladas corretamente.")

    print("\n" + "="*40)
    print("O modelo ONNX está pronto para ser usado em motores de inferência como")
    print("ONNX Runtime, TensorRT, ou para integração com TLLM.")
    print("="*40 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de Exportação de Modelos YOLO para ONNX")

    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help="Caminho para o arquivo de pesos do modelo treinado (ex: 'runs/train/train_run/weights/best.pt')."
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Tamanho das imagens de entrada (altura e largura) para o qual o modelo será exportado.'
    )
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Ativa a simplificação do modelo ONNX com onnx-simplifier.'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=12,
        help='Versão do ONNX opset a ser usada para a exportação. Padrão é 12.'
    )

    args = parser.parse_args()
    main(args)

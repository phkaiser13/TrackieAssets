import argparse
import os
import torch
from pathlib import Path
from transformers import DPTForDepthEstimation
from transformers.onnx import export, FeaturesManager
from onnxruntime.quantization import quantize_dynamic, QuantType

def main(args):
    """
    Main function to export a fine-tuned DPT model to ONNX and optionally quantize it.
    """
    # --- 1. Load the fine-tuned model ---
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"Diretório do modelo não encontrado em: {args.model_path}")

    print(f"Carregando modelo fine-tuned de: {args.model_path}")
    model = DPTForDepthEstimation.from_pretrained(args.model_path)
    model.eval() # Set model to evaluation mode

    # --- 2. Export to ONNX (FP32) ---
    # Define the output path for the standard ONNX model
    output_onnx_path = Path(args.output_path)
    os.makedirs(output_onnx_path.parent, exist_ok=True)

    # The feature determines the model's signature (inputs/outputs)
    feature = "depth-estimation"
    try:
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
        onnx_config = model_onnx_config(model.config)

        print("\n" + "="*40)
        print("Iniciando a Exportação para ONNX (FP32)")
        print(f"   - Modelo de Entrada: {args.model_path}")
        print(f"   - Arquivo de Saída: {output_onnx_path}")
        print("="*40 + "\n")

        # The export function saves the model to the specified path
        _ = export(
            preprocessor=model.processor,
            model=model,
            config=onnx_config,
            opset=onnx_config.default_opset,
            output=output_onnx_path
        )
        print(f"Modelo exportado com sucesso para: {output_onnx_path}")

    except Exception as e:
        print(f"Ocorreu um erro durante a exportação para ONNX: {e}")
        return

    # --- 3. Quantize to INT8 (optional) ---
    if args.quantize:
        print("\n" + "="*40)
        print("Iniciando Quantização para ONNX (INT8)")
        print("="*40 + "\n")

        output_quantized_path = output_onnx_path.with_name(output_onnx_path.stem + "_int8.onnx")

        try:
            quantize_dynamic(
                model_input=output_onnx_path,
                model_output=output_quantized_path,
                weight_type=QuantType.QInt8
            )
            print(f"Modelo quantizado com sucesso para: {output_quantized_path}")
            print("Este modelo INT8 é ideal para inferência rápida em CPUs e GPUs compatíveis.")

        except Exception as e:
            print(f"Ocorreu um erro durante a quantização: {e}")
            print("Verifique se a biblioteca 'onnxruntime' está instalada corretamente.")

    print("\n" + "="*40)
    print("Processo de exportação concluído.")
    print("="*40 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script para Exportar e Quantizar Modelos DPT (MiDaS)")

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help="Caminho para o diretório do modelo DPT fine-tuned (ex: './models/midas_finetuned')."
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='./models/onnx/dpt_swinv2_tiny_256.onnx',
        help='Caminho para salvar o arquivo ONNX exportado.'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Ativa a quantização dinâmica para INT8 após a exportação para ONNX.'
    )

    args = parser.parse_args()
    main(args)

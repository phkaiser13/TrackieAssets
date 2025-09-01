import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path

from common.config.parser import load_config

def validate_paths(config):
    """
    Checks if the necessary directories and files exist before starting.
    """
    piper_train_script = Path(config.piper_training_dir) / "src" / "python" / "train.py"
    if not piper_train_script.is_file():
        print(f"ERRO: Script de treinamento do Piper não encontrado em: {piper_train_script}")
        print("Por favor, clone o repositório 'piper-training' e verifique o caminho em 'piper_training_dir' no seu arquivo de configuração.")
        sys.exit(1)

    dataset_dir = Path(config.dataset.dir)
    if not dataset_dir.is_dir():
        print(f"ERRO: Diretório do dataset não encontrado em: {dataset_dir}")
        sys.exit(1)

    metadata_path = dataset_dir / "metadata.csv"
    wavs_path = dataset_dir / "wavs"
    if not metadata_path.is_file() or not wavs_path.is_dir():
        print(f"ERRO: O dataset não parece estar no formato LJSpeech.")
        print(f"Esperado: {metadata_path} e {wavs_path}")
        sys.exit(1)

    if config.base_model_path and not Path(config.base_model_path).is_file():
        print(f"AVISO: Arquivo do modelo base não encontrado em: {config.base_model_path}. O treinamento começará do zero.")

def find_final_model(piper_output_dir: Path) -> (Path, Path):
    """
    Finds the paths to the generated .onnx and .json files in the last epoch's directory.
    """
    last_epoch_dir = None
    last_epoch = -1

    for epoch_dir in piper_output_dir.glob("epoch_*"):
        if epoch_dir.is_dir():
            try:
                epoch_num = int(epoch_dir.name.split('_')[1])
                if epoch_num > last_epoch:
                    last_epoch = epoch_num
                    last_epoch_dir = epoch_dir
            except (ValueError, IndexError):
                continue

    if last_epoch_dir:
        onnx_file = next(last_epoch_dir.glob("*.onnx"), None)
        json_file = next(last_epoch_dir.glob("*.json"), None)
        if onnx_file and json_file:
            return onnx_file, json_file

    return None, None


def main(config_path: str):
    """
    Main function to orchestrate the Piper TTS fine-tuning process.
    """
    # 1. Load Configuration
    config = load_config(config_path)
    print("Configuração de treinamento TTS carregada.")

    # 2. Validate Paths
    validate_paths(config)
    print("Caminhos validados. O script de treinamento e o dataset foram encontrados.")

    # 3. Construct the Piper Training Command
    piper_train_script = Path(config.piper_training_dir) / "src" / "python" / "train.py"
    # The Piper script saves its output relative to its own location,
    # so we define an output path inside the piper-training dir.
    internal_output_dir = Path(config.piper_training_dir) / "run_output"

    cmd = [
        sys.executable, str(piper_train_script),
        "--dataset-dir", str(Path(config.dataset.dir).absolute()),
        "--output-dir", str(internal_output_dir.absolute()),
        "--accelerator", config.hardware.accelerator,
        "--devices", str(config.hardware.devices),
        "--precision", config.hardware.precision,
        "--max_epochs", str(config.training_params.epochs),
        "--batch-size", str(config.training_params.batch_size),
        "--learning_rate", str(config.training_params.learning_rate),
        "--quality", config.training_params.quality,
        "--validation-split", str(config.dataset.validation_split),
        "--speaker_id", str(config.phonemization.speaker_id),
    ]

    if config.base_model_path:
        cmd.extend(["--resume_from_checkpoint", str(Path(config.base_model_path).absolute())])

    print("\n" + "="*40)
    print("Iniciando Fine-Tuning do Piper TTS")
    print(f"   - Comando a ser executado: {' '.join(cmd)}")
    print("="*40 + "\n")

    # 4. Execute the Training Command
    try:
        # We use Popen to stream the output in real-time
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        rc = process.poll()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)

        print("\nTreinamento do Piper concluído com sucesso!")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\nERRO: A execução do script de treinamento do Piper falhou: {e}")
        sys.exit(1)

    # 5. Copy Artifacts to Final Destination
    print("Procurando pelo modelo treinado...")
    onnx_file, json_file = find_final_model(internal_output_dir)

    if onnx_file and json_file:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(onnx_file, output_dir)
        shutil.copy(json_file, output_dir)

        print("\n" + "="*40)
        print("Orquestração concluída!")
        print(f"Modelo ONNX e JSON copiados para: {output_dir.absolute()}")
        print("="*40 + "\n")
    else:
        print("\nAVISO: Não foi possível encontrar o modelo .onnx final no diretório de saída do Piper.")
        print(f"Verifique o diretório '{internal_output_dir}' para os artefatos de treinamento.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de orquestração para Fine-Tuning de TTS com Piper")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/tts_finetune.yml',
        help='Caminho para o arquivo de configuração YAML.'
    )
    args = parser.parse_args()
    main(args.config)

import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
import os

from common.config.parser import load_config

def get_best_device():
    """
    Detects and returns the best available computing device.
    Prioritizes CUDA > MPS > CPU. For LLM QLoRA, CUDA is strongly recommended.
    """
    if torch.cuda.is_available():
        print("CUDA disponível. Usando GPU.")
        return "cuda"
    # Note: MPS support for 4-bit/8-bit training is limited or non-existent.
    # We check for it but QLoRA will likely fail.
    elif torch.backends.mps.is_available():
        print("Apple MPS (Metal) disponível. Usando MPS.")
        print("AVISO: O treinamento QLoRA pode não ser compatível com MPS.")
        return "mps"
    else:
        print("CUDA e MPS não encontrados. Usando CPU (extremamente lento).")
        return "cpu"

def main(config_path: str):
    """
    Main function to run the LLM fine-tuning pipeline.
    """
    # 1. Load Configuration
    config = load_config(config_path)

    # 2. Setup Device
    if config.device == "auto":
        device = get_best_device()
    else:
        device = config.device

    if "cuda" not in device:
        print("AVISO: O treinamento quantizado (QLoRA) é otimizado para CUDA. Outros dispositivos podem falhar ou ser muito lentos.")
    if device == "rocm":
        print("AVISO: O suporte para ROCm com bitsandbytes pode ser experimental. Verifique a compatibilidade da sua versão.")

    # 3. Configure Quantization (BitsAndBytes)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.quantization.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
    )

    # 4. Load Tokenizer and Model
    print(f"Carregando modelo base: {config.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, trust_remote_code=True)
    # Set a padding token if one is not already set.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        quantization_config=quant_config,
        device_map="auto", # Automatically handle device placement
        trust_remote_code=True,
    )
    model.config.use_cache = False # Recommended for training

    # 5. Prepare Model for K-bit Training and Apply LoRA
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora.target_modules,
    )
    model = get_peft_model(model, lora_config)
    print("Modelo preparado com QLoRA.")
    model.print_trainable_parameters()

    # 6. Load Dataset
    print(f"Carregando dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, split=config.split)
    print(f"Dataset carregado com {len(dataset)} exemplos.")

    # 7. Configure Training Arguments
    training_args_dict = config.training_args._config
    trainer_args = TrainingArguments(**training_args_dict)

    # 8. Initialize SFT Trainer
    print("Inicializando o SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field=config.text_column,
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
        args=trainer_args,
        packing=True, # Packs multiple short examples into one sequence for efficiency
    )

    # 9. Start Fine-Tuning
    print("\n" + "="*40)
    print("Iniciando Fine-Tuning do LLM com QLoRA")
    print(f"   - Modelo: {config.base_model_name}")
    print(f"   - Dataset: {config.dataset_name}")
    print(f"   - Output Dir: {config.output_dir}")
    print("="*40 + "\n")

    trainer.train()

    # 10. Save the final adapter
    final_adapter_path = os.path.join(config.output_dir, "final_adapter")
    print(f"\nSalvando o adaptador LoRA final em: {final_adapter_path}")
    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)

    print("\n" + "="*40)
    print("Fine-tuning concluído com sucesso!")
    print(f"Adaptador LoRA salvo e pronto para inferência ou merge.")
    print("="*40 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de Fine-Tuning de LLM com QLoRA")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mistral_finetune.yml',
        help='Caminho para o arquivo de configuração YAML.'
    )
    args = parser.parse_args()
    main(args.config)

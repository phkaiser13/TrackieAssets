import torch
import argparse
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, Audio

from common.config.parser import load_config

# --- Data Collator ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Custom data collator that dynamically pads the input features and labels for seq2seq tasks.
    This is essential for efficient training on batches of varying lengths.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need
        # different padding methods.
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad the input features (log-Mel spectrograms)
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad the labels (token IDs)
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore it in the loss calculation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If the model is being trained with CTC, the padding value should be the tokenizer's pad_token_id
        if (
            hasattr(self.processor, "model_input_names")
            and "labels" in self.processor.model_input_names
        ):
            batch["labels"] = labels
        else:
            batch["labels"] = labels

        return batch

# --- Main Training Function ---
def main(config_path: str):
    """
    Main function to run the ASR (Whisper) fine-tuning pipeline.
    """
    # 1. Load Configuration
    config = load_config(config_path)

    # 2. Load Whisper Processor (Feature Extractor + Tokenizer)
    print(f"Carregando processador para o modelo: {config.model_name}")
    processor = WhisperProcessor.from_pretrained(
        config.model_name,
        language=config.processor.language,
        task=config.processor.task
    )

    # 3. Load and Prepare Dataset
    print(f"Carregando e preparando o dataset: {config.dataset.name} ({config.dataset.subset})")
    raw_dataset = load_dataset(
        config.dataset.name,
        config.dataset.subset,
        split={
            "train": config.dataset.train_split,
            "eval": config.dataset.val_split,
        },
        use_auth_token=True,
        streaming=False, # Set to True for very large datasets
    )

    # Resample audio to 16kHz as required by Whisper
    raw_dataset = raw_dataset.cast_column(config.dataset.audio_column, Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        # Process audio to log-Mel spectrogram
        audio = batch[config.dataset.audio_column]
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # Process text to token IDs
        batch["labels"] = processor.tokenizer(batch[config.dataset.text_column]).input_ids
        return batch

    print("Processando o dataset (pode levar um tempo)...")
    tokenized_dataset = raw_dataset.map(prepare_dataset, remove_columns=raw_dataset["train"].column_names)

    # 4. Initialize Data Collator, Model, and Metrics
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    print("Carregando o modelo pré-treinado...")
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    print("Carregando a métrica WER...")
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # 5. Configure Training Arguments
    training_args_dict = config.training_args._config
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # 6. Initialize Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # 7. Start Training
    print("\n" + "="*40)
    print("Iniciando Fine-Tuning do Whisper")
    print(f"   - Modelo: {config.model_name}")
    print(f"   - Dataset: {config.dataset.name}")
    print(f"   - Output Dir: {config.training_args.output_dir}")
    print("="*40 + "\n")

    trainer.train()

    # 8. Save the final model and processor
    print(f"\nSalvando o modelo final em: {config.output_dir}")
    model.save_pretrained(config.output_dir)
    processor.save_pretrained(config.output_dir)

    print("\n" + "="*40)
    print("Fine-tuning do ASR concluído com sucesso!")
    print("="*40 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de Fine-Tuning de ASR com Whisper")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/asr_finetune.yml',
        help='Caminho para o arquivo de configuração YAML.'
    )
    args = parser.parse_args()
    main(args.config)

import argparse
import os

# Importa todos os componentes que construímos
from audio_processing.pipeline import AudioOrchestrator
from inference_wrappers import WhisperCpp, LlamaCpp
from vision_processing.yolo_onnx_infer import run_yolo_inference, YOLO_CLASS_NAMES
from vision_processing.midas_onnx_infer import run_midas_inference

def build_llm_prompt(detections: list, depth_map_path: str) -> str:
    """
    Constrói um prompt detalhado para o LLM com base nos resultados da visão.
    """
    if not detections:
        return "Descreva uma cena, mas os modelos de visão não detectaram objetos específicos."

    # Contagem de objetos
    object_counts = {}
    for det in detections:
        label = det['label']
        object_counts[label] = object_counts.get(label, 0) + 1

    object_strings = [f"{count} {label}{'s' if count > 1 else ''}" for label, count in object_counts.items()]

    prompt = (
        "Você é um assistente de IA que descreve cenas a partir de dados de visão computacional.\n"
        "Com base nas seguintes informações, forneça uma descrição concisa e natural da cena em um único parágrafo:\n\n"
        f"- Objetos detectados: {', '.join(object_strings)}.\n"
        f"- Um mapa de profundidade visualizado foi gerado em '{depth_map_path}', onde áreas mais claras estão mais próximas.\n\n"
        "Descrição da cena:"
    )
    return prompt

def main(args):
    """
    O Orquestrador Geral que une todos os componentes de IA.
    """
    # --- 1. Inicialização dos Componentes ---
    print("--- Inicializando Pipeline de IA Multimodal ---")
    try:
        audio_orchestrator = AudioOrchestrator(
            wake_word=args.wake_word,
            silence_threshold=0.6
        )
        whisper = WhisperCpp(model_path=args.whisper_model)
        llama = LlamaCpp(model_path=args.llama_model, n_gpu_layers=args.n_gpu_layers, n_ctx=4096)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\nERRO FATAL na inicialização: {e}")
        return

    # --- 2. Loop Principal ---
    audio_orchestrator.say("Sistema de IA multimodal iniciado. Diga 'computador' para começar.")

    while True:
        try:
            # Espera pelo comando de voz e o grava
            command_audio_path = audio_orchestrator.listen_for_command()
            if not command_audio_path:
                print("Nenhum comando gravado. Reiniciando.")
                continue

            # Transcreve o comando de voz para texto
            transcribed_text = whisper.transcribe(command_audio_path).lower()
            print(f"Comando transcrito: '{transcribed_text}'")
            os.remove(command_audio_path) # Limpa o arquivo temporário

            # --- 3. Lógica de Decisão e Execução do Pipeline ---
            if "analise a imagem" in transcribed_text or "descreva a cena" in transcribed_text:
                if not os.path.exists(args.image_to_analyze):
                    audio_orchestrator.say(f"Desculpe, não encontrei a imagem para analisar em {args.image_to_analyze}")
                    continue

                audio_orchestrator.say("Entendido. Analisando a imagem agora. Isso pode levar um momento.")

                # Executa os modelos de visão
                yolo_detections = run_yolo_inference(args.yolo_model, args.image_to_analyze, YOLO_CLASS_NAMES)
                depth_map_path = run_midas_inference(args.midas_model, args.image_to_analyze)

                # Constrói o prompt e obtém a resposta do LLM
                prompt = build_llm_prompt(yolo_detections, depth_map_path)
                print(f"\n--- Prompt para o LLM ---\n{prompt}\n-------------------------\n")

                llm_response = llama.generate_response(prompt)
                print(f"Resposta do LLM: {llm_response}")

                # Fala a resposta
                audio_orchestrator.say(llm_response)

            elif "desligar" in transcribed_text or "encerrar" in transcribed_text:
                audio_orchestrator.say("Encerrando o sistema. Até logo!")
                break
            else:
                # Fallback: passa o comando diretamente para o LLM
                audio_orchestrator.say("Entendido. Deixe-me pensar sobre isso.")
                llm_response = llama.generate_response(transcribed_text)
                print(f"Resposta do LLM: {llm_response}")
                audio_orchestrator.say(llm_response)

        except KeyboardInterrupt:
            print("\nEncerrando o sistema...")
            audio_orchestrator.say("Sistema encerrado.")
            break
        except Exception as e:
            print(f"Ocorreu um erro no loop principal: {e}")
            audio_orchestrator.say("Ocorreu um erro. Por favor, verifique o console.")
            time.sleep(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Orquestrador Geral de IA Multimodal")

    # --- Argumentos dos Modelos ---
    parser.add_argument('--yolo-model', type=str, default='./models/onnx/yolov5nu.onnx')
    parser.add_argument('--midas-model', type=str, default='./models/onnx/dpt_swinv2_tiny_256_int8.onnx')
    parser.add_argument('--whisper-model', type=str, required=True, help="Caminho para o modelo Whisper.cpp (ex: './whisper.cpp/models/ggml-tiny.en.bin')")
    parser.add_argument('--llama-model', type=str, required=True, help="Caminho para o modelo Llama.cpp (ex: './models/mistral-7b.gguf')")
    parser.add_argument('--image-to-analyze', type=str, default='./tests-data/sample_image.jpg')

    # --- Argumentos de Hardware e Pipeline ---
    parser.add_argument('--n-gpu-layers', type=int, default=35, help="Número de camadas do LLM para descarregar na GPU.")
    parser.add_argument('--wake-word', type=str, default='computer', help="Palavra de ativação para o sistema.")

    args = parser.parse_args()
    main(args)

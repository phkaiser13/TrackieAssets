import os
import pvporcupine
import struct
import sounddevice as sd
import numpy as np
import torch
import wave
import time
from piper import PiperVoice
from pathlib import Path
import platform
import requests
import zipfile
import io

# --- Constantes e Configurações ---
PICOVOICE_ACCESS_KEY = "COLOQUE_SUA_CHAVE_DE_ACESSO_AQUI" # OBRIGATÓRIO: Obtenha em https://console.picovoice.ai/
TEMP_RECORDING_PATH = "temp_user_command.wav"

class AudioOrchestrator:
    """
    Orquestra o pipeline de áudio: Wake Word -> VAD -> Gravação -> TTS.
    """
    def __init__(self, wake_word="porcupine", silence_threshold=0.5, silence_duration=1.5):
        """
        Inicializa o orquestrador de áudio.
        """
        if PICOVOICE_ACCESS_KEY == "COLOQUE_SUA_CHAVE_DE_ACESSO_AQUI":
            raise ValueError("A chave de acesso da Picovoice (PICOVOICE_ACCESS_KEY) não foi definida.")

        # --- 1. Inicializar o Piper TTS ---
        self.tts_voice = self._initialize_piper_voice()

        # --- 2. Inicializar o Porcupine Wake Word Engine ---
        self.porcupine = self._initialize_porcupine(wake_word)

        # --- 3. Inicializar o Silero VAD ---
        self.vad_model, self.vad_utils = self._initialize_silero_vad()

        # --- 4. Configurações de áudio ---
        self.sample_rate = 16000  # VAD e Porcupine operam a 16kHz
        self.frame_length = self.porcupine.frame_length
        self.silence_threshold = silence_threshold
        self.silence_duration_frames = int((silence_duration * self.sample_rate) / self.frame_length)

    def _initialize_piper_voice(self, model_name="pt_BR-faber-medium.onnx"):
        """Inicializa e baixa a voz do Piper TTS se necessário."""
        model_path_str = f"tts_models/{model_name}"
        config_path_str = f"{model_path_str}.json"

        model_path = Path(model_path_str)
        config_path = Path(config_path_str)

        if not model_path.exists() or not config_path.exists():
            print(f"Modelo de voz '{model_name}' não encontrado. Baixando...")
            os.makedirs(model_path.parent, exist_ok=True)
            voice_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/{os.path.splitext(model_name)[0]}.zip"

            try:
                response = requests.get(voice_url)
                response.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    z.extractall(model_path.parent)
                print("Download e extração concluídos.")
            except Exception as e:
                raise RuntimeError(f"Falha ao baixar o modelo de voz: {e}")

        return PiperVoice.load(model_path_str, config_path_str)

    def _initialize_porcupine(self, wake_word):
        """Inicializa o motor de wake word Porcupine."""
        try:
            return pvporcupine.create(
                access_key=PICOVOICE_ACCESS_KEY,
                keyword_paths=[pvporcupine.KEYWORD_PATHS[wake_word]]
            )
        except Exception as e:
            raise RuntimeError(f"Falha ao inicializar o Porcupine: {e}. "
                             "Verifique sua chave de acesso e a palavra-chave.")

    def _initialize_silero_vad(self):
        """Inicializa o modelo Silero VAD a partir do Torch Hub."""
        try:
            torch.set_num_threads(1) # Otimização para VAD
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            return model, utils
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar o modelo Silero VAD: {e}")

    def say(self, text: str):
        """Sintetiza e reproduz o áudio de um texto."""
        print(f"TTS: {text}")
        output_file = "tts_output.wav"
        with wave.open(output_file, 'wb') as wf:
            self.tts_voice.synthesize(text, wf)

        # Reproduz o arquivo de áudio
        data, fs = sd.read(output_file, dtype='int16')
        sd.play(data, fs)
        sd.wait()
        os.remove(output_file)

    def listen_for_command(self) -> str or None:
        """
        Ouve o microfone, espera pela wake word, grava um comando e o retorna.
        """
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16',
            blocksize=self.frame_length
        )
        stream.start()
        print(f"\nOuvindo pela wake word '{self.porcupine.keywords[0]}'...")

        try:
            while True:
                # --- 1. Loop de detecção da Wake Word ---
                pcm = stream.read(self.frame_length)[0]
                keyword_index = self.porcupine.process(pcm)

                if keyword_index >= 0:
                    print("Wake word detectada!")
                    self.say("Sim?")

                    # --- 2. Loop de gravação com VAD ---
                    recorded_frames = []
                    silent_frames = 0
                    is_speaking = False

                    # Pequeno buffer inicial para não perder o início da fala
                    recorded_frames.append(pcm)

                    while True:
                        pcm_vad = stream.read(self.frame_length)[0]
                        recorded_frames.append(pcm_vad)

                        # Converte para o formato que o Silero VAD espera (float32)
                        audio_float32 = np.frombuffer(pcm_vad, dtype=np.int16).astype(np.float32) / 32768.0
                        speech_prob = self.vad_model(torch.from_numpy(audio_float32), self.sample_rate).item()

                        if speech_prob > self.silence_threshold:
                            is_speaking = True
                            silent_frames = 0
                        else:
                            if is_speaking:
                                silent_frames += 1
                                if silent_frames > self.silence_duration_frames:
                                    print("Fim da fala detectado. Processando...")
                                    break # Sai do loop de gravação

                    # Salva o áudio gravado em um arquivo temporário
                    with wave.open(TEMP_RECORDING_PATH, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(b''.join(struct.pack('h', sample) for frame in recorded_frames for sample in frame))

                    stream.stop()
                    return TEMP_RECORDING_PATH

        except KeyboardInterrupt:
            print("Encerrando...")
        finally:
            if self.porcupine:
                self.porcupine.delete()
            if stream.active:
                stream.stop()
                stream.close()

if __name__ == '__main__':
    # Exemplo de como usar o orquestrador
    try:
        orchestrator = AudioOrchestrator(wake_word="computer")

        # Demonstração do TTS
        orchestrator.say("Olá! Eu estou pronta. Diga 'computer' para me ativar.")

        # Inicia o loop de escuta
        command_audio_path = orchestrator.listen_for_command()

        if command_audio_path:
            print(f"Comando de áudio gravado em: {command_audio_path}")
            # Aqui, você passaria o `command_audio_path` para o wrapper do Whisper
            orchestrator.say("Entendido. Processando seu comando.")
            # Ex: transcribed_text = whisper.transcribe(command_audio_path)

    except (ValueError, RuntimeError) as e:
        print(f"ERRO: {e}")
    except KeyboardInterrupt:
        print("Programa encerrado pelo usuário.")

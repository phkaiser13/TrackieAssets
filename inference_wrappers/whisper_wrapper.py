import subprocess
import os
import shlex
from pathlib import Path

class WhisperCpp:
    """
    Um wrapper Python para o executável Whisper.cpp.

    Esta classe gerencia a chamada ao executável `main` do Whisper.cpp
    para realizar a transcrição de arquivos de áudio.
    """
    def __init__(self, model_path: str, whisper_cpp_dir: str = './whisper.cpp', **kwargs):
        """
        Inicializa o wrapper WhisperCpp.

        Args:
            model_path (str): Caminho para o arquivo do modelo Whisper no formato GGML.
            whisper_cpp_dir (str): Caminho para o dir onde o whisper.cpp foi compilado.
            **kwargs: Argumentos adicionais para a transcrição.
                      Ex: language='en', threads=6
        """
        self.executable_path = os.path.join(whisper_cpp_dir, 'main')
        self.model_path = model_path

        if not os.path.exists(self.executable_path):
            raise FileNotFoundError(f"Executável do Whisper.cpp não encontrado em: {self.executable_path}. "
                                    "Por favor, execute o script 'setup_inference_engines.sh'.")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Arquivo do modelo não encontrado em: {self.model_path}.")

        # Parâmetros padrão, podem ser sobrescritos por kwargs
        self.params = {
            "language": "en",  # Idioma do áudio. 'auto' para detecção automática.
            "threads": 6,      # Número de threads para usar.
            "output_txt": True # Força a saída em formato de texto simples.
        }
        self.params.update(kwargs)

    def transcribe(self, audio_filepath: str, **kwargs) -> str:
        """
        Transcreve um arquivo de áudio para texto.

        Args:
            audio_filepath (str): Caminho para o arquivo de áudio (ex: .wav, .mp3).
            **kwargs: Permite sobrescrever temporariamente os parâmetros de transcrição.

        Returns:
            str: O texto transcrito.
        """
        if not os.path.exists(audio_filepath):
            raise FileNotFoundError(f"Arquivo de áudio não encontrado em: {audio_filepath}")

        current_params = self.params.copy()
        current_params.update(kwargs)

        # Usamos -f para o arquivo de áudio de entrada
        command = [self.executable_path, "-m", self.model_path, "-f", audio_filepath]

        for key, value in current_params.items():
            cli_arg = "--" + key.replace("_", "-")
            # Flags booleanas como 'output_txt' não precisam de valor se forem True
            if isinstance(value, bool) and value:
                command.append(cli_arg)
            elif not isinstance(value, bool):
                command.extend([cli_arg, str(value)])

        print(f"Executando Whisper.cpp com o comando: {' '.join(shlex.quote(c) for c in command)}")

        try:
            # Whisper.cpp com -otxt imprime o resultado limpo no stdout.
            # Se -otxt for usado, ele também cria um arquivo .txt.
            # A captura do stdout é geralmente mais limpa para um wrapper.
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding='utf-8',
                check=True
            )

            # A saída de texto limpa vai para stdout
            transcribed_text = process.stdout.strip()

            # Limpar o arquivo .txt gerado se o modo de saída de texto foi usado
            if current_params.get("output_txt"):
                expected_txt_file = Path(audio_filepath).with_suffix('.wav.txt')
                if expected_txt_file.exists():
                    os.remove(expected_txt_file)

            return transcribed_text

        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar Whisper.cpp:")
            print(f"Stderr: {e.stderr}")
            raise RuntimeError(f"Whisper.cpp falhou com o código de saída {e.returncode}") from e
        except Exception as e:
            print(f"Ocorreu um inesperado: {e}")
            raise e

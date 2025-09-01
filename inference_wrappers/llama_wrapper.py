import subprocess
import os
import shlex

class LlamaCpp:
    """
    Um wrapper Python para o executável Llama.cpp.

    Esta classe gerencia a chamada ao executável `main` do Llama.cpp,
    passando os parâmetros apropriados e capturando a saída.
    """
    def __init__(self, model_path: str, llama_cpp_dir: str = './llama.cpp', **kwargs):
        """
        Inicializa o wrapper LlamaCpp.

        Args:
            model_path (str): Caminho para o arquivo do modelo no formato GGUF.
            llama_cpp_dir (str): Caminho para o diretório onde o llama.cpp foi compilado.
            **kwargs: Argumentos adicionais para a geração de texto.
                      Ex: n_gpu_layers=35, n_ctx=2048, temp=0.7
        """
        self.executable_path = os.path.join(llama_cpp_dir, 'main')
        self.model_path = model_path

        if not os.path.exists(self.executable_path):
            raise FileNotFoundError(f"Executável do Llama.cpp não encontrado em: {self.executable_path}. "
                                    "Por favor, execute o script 'setup_inference_engines.sh'.")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Arquivo do modelo não encontrado em: {self.model_path}.")

        # Parâmetros padrão para a geração, podem ser sobrescritos por kwargs
        self.params = {
            "n_gpu_layers": 35, # Número de camadas para descarregar na GPU. Essencial para performance.
            "n_ctx": 4096,      # Tamanho do contexto.
            "n_predict": 512,   # Número de tokens a serem gerados.
            "temp": 0.8,        # Temperatura da amostragem.
            "top_p": 0.95,      # Amostragem Top-p.
            "threads": 6,       # Número de threads para usar na CPU.
        }
        self.params.update(kwargs)

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Gera uma resposta de texto a partir de um prompt.

        Args:
            prompt (str): O texto de entrada para o modelo.
            **kwargs: Permite sobrescrever temporariamente os parâmetros de geração.

        Returns:
            str: A resposta de texto gerada pelo modelo.
        """
        # Atualiza os parâmetros com quaisquer argumentos específicos desta chamada
        current_params = self.params.copy()
        current_params.update(kwargs)

        command = [self.executable_path, "-m", self.model_path, "-p", prompt]

        for key, value in current_params.items():
            # Converte a chave Python (ex: n_gpu_layers) para o formato de linha de comando (ex: --n-gpu-layers)
            cli_arg = "--" + key.replace("_", "-")
            command.extend([cli_arg, str(value)])

        print(f"Executando Llama.cpp com o comando: {' '.join(shlex.quote(c) for c in command)}")

        try:
            # `capture_output=True` para capturar stdout/stderr
            # `text=True` para decodificar como texto
            # `encoding='utf-8'` para evitar problemas de caracteres
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding='utf-8',
                check=True # Lança exceção se o processo retornar um código de erro
            )

            # A saída do Llama.cpp inclui o prompt. Nós queremos apenas a resposta gerada.
            # A resposta começa logo após o prompt ter sido impresso no stdout.
            full_output = process.stdout
            response_start_index = full_output.find(prompt) + len(prompt)
            generated_text = full_output[response_start_index:].strip()

            return generated_text

        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar Llama.cpp:")
            print(f"Stderr: {e.stderr}")
            raise RuntimeError(f"Llama.cpp falhou com o código de saída {e.returncode}") from e
        except Exception as e:
            print(f"Ocorreu um erro inesperado: {e}")
            raise e

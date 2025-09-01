# Estrutura do Conjunto de Dados Customizado para Fine-Tuning do MiDaS (DPT)

Este diretório está preparado para receber seu conjunto de dados customizado para o fine-tuning de modelos de estimativa de profundidade monocular como o MiDaS.

## Como Popular este Diretório

O fine-tuning requer pares de imagens RGB e seus respectivos mapas de profundidade.

1.  **Dados de Treinamento:**
    *   Coloque suas imagens RGB de **treinamento** no diretório `train/rgb/`.
    *   Coloque os mapas de profundidade correspondentes em `train/depth/`.

2.  **Dados de Validação:**
    *   Coloque suas imagens RGB de **validação** no diretório `val/rgb/`.
    *   Coloque os mapas de profundidade de validação correspondentes em `val/depth/`.

## Requisitos de Nomenclatura e Formato

-   **Nomes de Arquivos Correspondentes:** É **essencial** que cada imagem RGB tenha um mapa de profundidade com o **exatamente o mesmo nome de arquivo**.
    -   Exemplo: `train/rgb/scene1_frame001.jpg` deve corresponder a `train/depth/scene1_frame001.png`.

-   **Formato das Imagens RGB:** Formatos padrão como `.jpg` ou `.png` são aceitáveis.

-   **Formato dos Mapas de Profundidade:**
    -   Recomendado: **PNG de 16 bits, canal único (grayscale)**. Este formato preserva bem a precisão da profundidade.
    -   Alternativa: Arquivos `.npy` (Numpy) contendo o array de profundidade. O script de treinamento precisaria de um pequeno ajuste para carregar `.npy` em vez de imagens.
    -   Os valores no mapa de profundidade geralmente representam a distância em uma escala (por exemplo, milímetros) ou são valores de profundidade inversa. O script de treinamento normalizará esses dados.

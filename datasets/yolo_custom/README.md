# Estrutura do Conjunto de Dados Customizado para YOLO

Este diretório está preparado para receber seu conjunto de dados customizado para treinar modelos YOLO.

## Como Popular este Diretório

1.  **Imagens de Treinamento:**
    *   Coloque todas as suas imagens de **treinamento** no diretório `images/train/`.
    *   Formatos suportados: `.jpg`, `.png`, etc.

2.  **Imagens de Validação:**
    *   Coloque todas as suas imagens de **validação** no diretório `images/val/`.
    *   É crucial ter um conjunto de validação separado para avaliar a performance do modelo de forma não enviesada.

3.  **Arquivos de Anotação (Labels):**
    *   Para cada imagem, deve haver um arquivo de texto (`.txt`) correspondente na pasta `labels`.
    *   Exemplo: para `images/train/carro1.jpg`, deve existir `labels/train/carro1.txt`.
    *   O nome do arquivo de anotação DEVE ser o mesmo da imagem, apenas com a extensão trocada para `.txt`.

## Formato do Arquivo de Anotação (`.txt`)

Cada arquivo `.txt` deve conter uma linha para cada objeto (bounding box) na imagem correspondente.

O formato de cada linha é:
`<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>`

Onde:
*   `<class_id>`: É um número inteiro (começando em 0) que representa a classe do objeto. A correspondência entre o ID e o nome da classe é definida no arquivo `.yaml` de dados (ex: `yolo_custom_data.yaml`).
*   `<x_center_norm>`, `<y_center_norm>`: São as coordenadas do centro do bounding box, normalizadas para estarem entre 0 e 1 (divididas pela largura e altura da imagem, respectivamente).
*   `<width_norm>`, `<height_norm>`: São a largura e a altura do bounding box, também normalizadas para estarem entre 0 e 1.

### Exemplo

Para uma imagem `imagem1.jpg` com um "carro" (classe 0) e uma "pessoa" (classe 1), o arquivo `imagem1.txt` poderia ter o seguinte conteúdo:

```
0 0.54 0.62 0.3 0.25
1 0.21 0.45 0.15 0.5
```

import cv2
import numpy as np
import onnxruntime as ort
import os

def preprocess_midas(image, input_size=(256, 256)):
    """Pre-processa a imagem para o formato de entrada do MiDaS DPT."""
    # O processador do DPT redimensiona e normaliza.
    # Aqui replicamos o comportamento essencial.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image_rgb, input_size)

    # Normalização (valores de [0, 255] para [0, 1])
    image_data = resized_image.astype('float32') / 255.0

    # Transposição para CHW e adição da dimensão do lote
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)

    return image_data

def postprocess_midas(depth_map, original_size):
    """Post-processa a saída do MiDaS para visualização."""
    # Remove a dimensão do lote e do canal
    depth_map = np.squeeze(depth_map)

    # Redimensiona para o tamanho da imagem original
    h_orig, w_orig = original_size
    resized_depth = cv2.resize(depth_map, (w_orig, h_orig))

    # Normaliza o mapa de profundidade para o intervalo [0, 255] para visualização
    depth_min = resized_depth.min()
    depth_max = resized_depth.max()

    if depth_max - depth_min > 1e-6:
        # Normalização para o intervalo 0-1
        display_map = (resized_depth - depth_min) / (depth_max - depth_min)
    else:
        display_map = np.zeros(resized_depth.shape, dtype=np.float32)

    # Converte para 8-bit e aplica um mapa de cores
    display_map = (display_map * 255).astype(np.uint8)
    display_map = cv2.applyColorMap(display_map, cv2.COLORMAP_INFERNO)

    return display_map

def run_midas_inference(onnx_path, image_path, output_image_path="midas_output.jpg"):
    """
    Executa a inferência de estimativa de profundidade em uma imagem usando um modelo ONNX.
    """
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem em: {image_path}")

    original_size = image.shape[:2]

    # O tamanho de entrada pode variar dependendo do modelo DPT exportado.
    # O modelo 'dpt-swinv2-tiny-256' espera 256x256.
    input_shape = session.get_inputs()[0].shape
    input_size = (input_shape[2], input_shape[3]) # (height, width)

    processed_image = preprocess_midas(image, input_size)

    # Executa a inferência
    result = session.run(None, {input_name: processed_image})

    # O resultado é o mapa de profundidade previsto
    predicted_depth = result[0]

    # Pós-processamento para visualização
    visualized_depth_map = postprocess_midas(predicted_depth, original_size)

    cv2.imwrite(output_image_path, visualized_depth_map)
    print(f"Mapa de profundidade visualizado salvo em: {output_image_path}")

    return output_image_path

if __name__ == '__main__':
    # Exemplo de uso
    # Supondo que o modelo está em ./models/onnx/
    # Usamos o modelo INT8 que é mais rápido para inferência
    midas_model_path = './models/onnx/dpt_swinv2_tiny_256_int8.onnx'
    test_image = './tests-data/sample_image.jpg' # Use a mesma imagem de teste

    if not os.path.exists(midas_model_path):
        print(f"ERRO: Modelo MiDaS não encontrado em {midas_model_path}. "
              "Certifique-se de executar o script de exportação do MiDaS com a flag --quantize.")
    elif not os.path.exists(test_image):
        print(f"ERRO: Imagem de teste não encontrada em {test_image}. Crie uma para executar este exemplo.")
    else:
        depth_map_path = run_midas_inference(midas_model_path, test_image)
        print(f"\nInferência do MiDaS concluída. Mapa de profundidade salvo em: {depth_map_path}")

        # Mostra a imagem resultante (requer ambiente com GUI)
        # try:
        #     result_img = cv2.imread(depth_map_path)
        #     cv2.imshow("Depth Map", result_img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # except cv2.error:
        #     print("\nNão foi possível exibir a imagem (ambiente sem GUI).")

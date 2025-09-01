import cv2
import numpy as np
import onnxruntime as ort

def preprocess_yolo(image, input_size=(640, 640)):
    """Pre-processa a imagem para o formato de entrada do YOLO."""
    h, w, _ = image.shape
    # Redimensiona mantendo a proporção com padding
    scale = min(input_size[0] / h, input_size[1] / w)
    resized_h, resized_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (resized_w, resized_h))

    # Cria um canvas e cola a imagem redimensionada
    padded_image = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    padded_image[(input_size[0] - resized_h) // 2:(input_size[0] - resized_h) // 2 + resized_h,
                 (input_size[1] - resized_w) // 2:(input_size[1] - resized_w) // 2 + resized_w, :] = resized_image

    # Normaliza e converte para NCHW
    image_data = np.array(padded_image, dtype='float32') / 255.0
    image_data = np.transpose(image_data, (2, 0, 1)) # HWC to CHW
    image_data = np.expand_dims(image_data, axis=0) # Add batch dimension

    return image_data, scale, (resized_w, resized_h)

def postprocess_yolo(output, confidence_thresh=0.5, nms_thresh=0.45, input_size=(640, 640), original_size=(1080, 1920)):
    """Post-processa a saída bruta do modelo YOLO."""
    # A saída do Ultralytics ONNX é [1, 84, 8400] onde 84 = 4 (box) + 80 (classes)
    # Transpomos para [1, 8400, 84] para facilitar o processamento
    output = np.transpose(output[0], (1, 0))

    boxes, scores, class_ids = [], [], []

    # Filtra por confiança
    max_scores = np.max(output[:, 4:], axis=1)
    mask = max_scores > confidence_thresh
    filtered_output = output[mask]

    if not filtered_output.any():
        return [], [], []

    # Extrai caixas, scores e classes
    for detection in filtered_output:
        box = detection[:4]
        score = np.max(detection[4:])
        class_id = np.argmax(detection[4:])

        # Converte de (center_x, center_y, width, height) para (x1, y1, x2, y2)
        cx, cy, w, h = box
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        class_ids.append(class_id)

    # Aplica Non-Maximum Suppression (NMS)
    # O ONNX Runtime não tem NMS embutido, então usamos o do OpenCV
    indices = cv2.dnn.NMSBoxes(boxes, np.array(scores), confidence_thresh, nms_thresh)

    final_boxes, final_scores, final_class_ids = [], [], []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_scores.append(scores[i])
            final_class_ids.append(class_ids[i])

    # Escala as caixas de volta para o tamanho da imagem original
    h_orig, w_orig = original_size
    scale_orig = min(input_size[0] / h_orig, input_size[1] / w_orig)
    pad_x = (input_size[1] - w_orig * scale_orig) / 2
    pad_y = (input_size[0] - h_orig * scale_orig) / 2

    scaled_boxes = []
    for box in final_boxes:
        x1, y1, x2, y2 = box
        x1 = (x1 - pad_x) / scale_orig
        y1 = (y1 - pad_y) / scale_orig
        x2 = (x2 - pad_x) / scale_orig
        y2 = (y2 - pad_y) / scale_orig
        scaled_boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return scaled_boxes, final_scores, final_class_ids

def run_yolo_inference(onnx_path, image_path, class_names, output_image_path="yolo_output.jpg"):
    """
    Executa a inferência YOLO em uma imagem usando um modelo ONNX.
    """
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    image = cv2.imread(image_path)
    original_size = image.shape[:2]

    processed_image, _, _ = preprocess_yolo(image)

    # Executa a inferência
    result = session.run(None, {input_name: processed_image})

    # Pós-processamento
    boxes, scores, class_ids = postprocess_yolo(result[0], original_size=original_size)

    detections = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        detections.append({
            "label": class_names[class_id],
            "confidence": float(score),
            "box": box
        })

        # Desenha na imagem
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[class_id]}: {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_image_path, image)
    print(f"Imagem com detecções salva em: {output_image_path}")

    return detections

if __name__ == '__main__':
    # Exemplo de uso
    YOLO_CLASS_NAMES = ['person', 'car', 'traffic_light', 'truck', 'bicycle'] # Exemplo

    # Supondo que os modelos estão em ./models/onnx/
    yolo_model_path = './models/onnx/yolov5nu.onnx' # Caminho para seu modelo treinado
    test_image = './tests-data/sample_image.jpg' # Crie esta imagem para teste

    if not os.path.exists(yolo_model_path):
        print(f"ERRO: Modelo YOLO não encontrado em {yolo_model_path}")
    elif not os.path.exists(test_image):
        print(f"ERRO: Imagem de teste não encontrada em {test_image}. Crie uma para executar este exemplo.")
    else:
        detected_objects = run_yolo_inference(yolo_model_path, test_image, YOLO_CLASS_NAMES)
        print("\nObjetos Detectados:")
        print(detected_objects)

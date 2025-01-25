from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
from ultralytics import YOLO
from collections import Counter
import base64
import numpy as np
import json

# Глобальные данные для анализа
global_boxes_data = []

def find_working_camera():
    """Находит первую работающую камеру."""
    for index in range(10):  # Проверяем индексы от 0 до 9
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Камера найдена с индексом: {index}")
            return cap
        cap.release()
    print("Нет доступных камер")
    return None

def gen_frames_yolo():
    """Генерация видеопотока с использованием YOLO."""
    global global_boxes_data
    cap = find_working_camera()  # Поиск работающей камеры

    if cap is None:
        raise RuntimeError("Не удалось найти работающую камеру")

    # Загружаем YOLO модель
    model = YOLO('yolov8n.pt')

    model_names = list(model.names.values())  # Классы модели

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Обрабатываем кадр с помощью YOLO
        results = model.predict(frame, conf=0.25)

        boxes_combined = []
        if len(results) > 0:
            boxes = results[0].boxes  # Получаем bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Координаты
                cls_id = int(box.cls[0])  # ID класса
                conf = float(box.conf[0])  # Уверенность
                class_name = model_names[cls_id] if cls_id < len(model_names) else "Unknown"

                boxes_combined.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': conf,
                    'class_name': class_name,
                })

        # Обновляем данные для анализа
        global_boxes_data = boxes_combined

        # Рисуем результаты на кадре
        for box in boxes_combined:
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
            class_name = box['class_name']
            confidence = box['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Кодируем кадр как JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def video_feed_yolo(request):
    """Реальный видеопоток с YOLO."""
    return StreamingHttpResponse(
        gen_frames_yolo(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def analysis(request):
    """Анализ обнаруженных объектов."""
    global global_boxes_data

    # Подсчет количества каждого класса
    class_counts = Counter(box['class_name'] for box in global_boxes_data)

    # Рассчитываем среднюю уверенность для каждого класса
    class_confidences = {label: [] for label in class_counts.keys()}
    for box in global_boxes_data:
        class_confidences[box['class_name']].append(box['confidence'])

    class_data = []
    for class_name, count in class_counts.items():
        avg_confidence = sum(class_confidences[class_name]) / count
        class_data.append({
            'class_name': class_name,
            'count': count,
            'avg_confidence': avg_confidence
        })

    context = {
        'class_data': class_data,
    }
    return render(request, 'analytics.html', context)

@csrf_exempt
def process_frame(request):
    """Обработка изображения, отправленного с клиента."""
    if request.method == 'POST':
        data = json.loads(request.body)
        frame_data = data.get('frame')

        # Декодируем изображение из base64
        img_data = base64.b64decode(frame_data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # YOLO
        model = YOLO('yolov8n.pt')
        results = model.predict(frame)

        detected_objects = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                detected_objects.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'class_name': model.names[cls_id],
                    'confidence': conf,
                })

        return JsonResponse({'detected_objects': detected_objects})

    return JsonResponse({'error': 'Invalid request'}, status=400)

def index(request):
    """Главная страница."""
    return render(request, 'home.html')

def solution(request):
    """Тестовая страница."""
    return render(request, 'solution.html')

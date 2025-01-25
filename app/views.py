from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
from ultralytics import YOLO
from collections import Counter
import base64
import numpy as np
import imageio.v3 as iio
import json


# Глобальные данные для анализа
global_boxes_data = []

# def gen_frames_yolo():
#     # Варианты источников видео (выберите один):
#     video_source = "example.mp4"  # Путь к видеофайлу
    
#     # Загрузка модели YOLO
#     model = YOLO("yolov8n.pt")
    
#     # Создание генератора кадров
#     for frame in iio.imiter(video_source, plugin="pyav"):
#         # Преобразование кадра в формат, подходящий для YOLO
#         yolo_frame = np.array(frame)
        
#         # Обработка кадра с помощью YOLO
#         results = model.predict(yolo_frame, conf=0.5)
#         annotated_frame = results[0].plot()  # Визуализация результатов
        
#         # Конвертация кадра в JPEG
#         _, buffer = cv2.imencode('.jpg', annotated_frame)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# def video_feed(request):
#     return StreamingHttpResponse(
#         gen_frames_yolo(), 
#         content_type='multipart/x-mixed-replace; boundary=frame'
#     )
def gen_frames_yolo():
    """Генерация видеопотока с использованием YOLO."""
    global global_boxes_data
    cap = cv2.VideoCapture("example.mp4") # Поиск работающей камеры

    if cap is None:
        raise RuntimeError("Не удалось найти работающую камеру")

    # Загружаем YOLO модель
    model1 = YOLO('yolov8n.pt')  # first model
    model4 = YOLO('sparkplug.pt')  # fourth model
    model5 = YOLO('pads.pt')  # fifth model
    model6 = YOLO('nutbolts.pt')  # sixth model
    model7 = YOLO('wheel.pt')  # seventh model

    model1_names = list(model1.names.values())
  
    model4_names = list(model4.names.values())
    model5_names = list(model5.names.values())
    model6_names = list(model6.names.values())
    model7_names = list(model7.names.values()) # Классы модели

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Обрабатываем кадр с помощью YOLO
        # Run predictions for all models
        results1 = model1.predict(frame, conf=0.25)

        results4 = model4.predict(frame, conf=0.25)
        results5 = model5.predict(frame, conf=0.8)
        results6 = model6.predict(frame, conf=0.8)
        results7 = model7.predict(frame, conf=0.6)

        boxes_combined = []
        for result, model_names in [
            (results1, model1_names),
           
            (results4, model4_names),
            (results5, model5_names),
            (results6, model6_names),
            (results7, model7_names)
            # Add other models here
        ]:
            if len(result) > 0:
                boxes = result[0].boxes  # bounding boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # coordinates
                    cls_id = int(box.cls[0])  # class index
                    conf = float(box.conf[0])  # confidence score
                    class_name = model_names[cls_id] if cls_id < len(model_names) else "Unknown"

                    # Convert confidence to integer percentage
                    
                
                    # Add to global_boxes_data
                    boxes_combined.append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'confidence': float(conf),
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
    """View to render the analysis page with detected objects."""
    global global_boxes_data
    # Count occurrences of each class
    class_counts = Counter(box['class_name'] for box in global_boxes_data)
    # Calculate average confidence for each class
    class_confidences = {label: [] for label in class_counts.keys()}
    for box in global_boxes_data:
        class_confidences[box['class_name']].append(box['confidence'])
    # Prepare data for each class
    class_data = []
    for class_name, count in class_counts.items():
        avg_confidence = sum(class_confidences[class_name]) / count
        class_data.append({
            'class_name': class_name,
            'count': count,
            'avg_confidence': avg_confidence
        })
    # Prepare data for charts
    labels = list(class_counts.keys())
    pie_data = list(class_counts.values())
    bar_data = [data['avg_confidence'] for data in class_data]
    context = {
        'labels': labels,
        'pie_data': pie_data,
        'bar_data': bar_data,
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

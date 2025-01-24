from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
import cv2
from ultralytics import YOLO
from collections import Counter


global_boxes_data = []
def gen_frames_yolo():
    global global_boxes_data
    cap = cv2.VideoCapture(0)

    # Load YOLO models (same as your current code)
    model1 = YOLO('yolov8n.pt')  # first model
    model2 = YOLO('calipers.pt')  # second model
    model3 = YOLO('lisenceplate.pt')  # third model
    model4 = YOLO('sparkplug.pt')  # fourth model
    model5 = YOLO('pads.pt')  # fifth model
    model6 = YOLO('nutbolts.pt')  # sixth model
    model7 = YOLO('wheel.pt')  # seventh model
    model8 = YOLO('headlights.pt')  # eighth model

    model1_names = list(model1.names.values())
    model2_names = list(model2.names.values())
    model3_names = list(model3.names.values())
    model4_names = list(model4.names.values())
    model5_names = list(model5.names.values())
    model6_names = list(model6.names.values())
    model7_names = list(model7.names.values())
    model8_names = list(model8.names.values())

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run predictions for all models
        results1 = model1.predict(frame, conf=0.25)
        results2 = model2.predict(frame, conf=0.6)
        results3 = model3.predict(frame, conf=0.8)
        results4 = model4.predict(frame, conf=0.25)
        results5 = model5.predict(frame, conf=0.8)
        results6 = model6.predict(frame, conf=0.8)
        results7 = model7.predict(frame, conf=0.6)
        results8 = model8.predict(frame, conf=0.25)
      

        boxes_combined = []
        for result, model_names in [
            (results1, model1_names),
            (results2, model2_names),
            (results3, model3_names),
            (results4, model4_names),
            (results5, model5_names),
            (results6, model6_names),
            (results7, model7_names),
            (results8, model8_names),
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

        # Update global variable for display
        global_boxes_data = boxes_combined

        # Draw bounding boxes on the frame
        for box in boxes_combined:
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
            class_name = box['class_name']
            confidence = box['confidence']/100
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence/100}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


def video_feed_yolo(request):
    """View for real-time video feed."""
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

def get_boxes_data(request):
    """API endpoint for fetching detected objects data and chart information."""
    global global_boxes_data

    return JsonResponse(global_boxes_data)
def test(request):
    """View for testing purposes."""
    return render(request, 'solution.html')


def index(request):
    """View for the home page."""
    return render(request, 'home.html')

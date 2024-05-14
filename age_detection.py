from cv2.typing import MatLike
from ultralytics import YOLO
import supervision as sv

model = YOLO(r'model_yolo/AgeDetection.pt')


def detect_age(image: MatLike) -> tuple:
    results = model(image)
    if not len(results[0]):
        return None, []
    detections = sv.Detections.from_ultralytics(results[0])

    # classes = model.names
    classes = [
        '4 -> 13',
        '14 -> 20',
        '21 -> 37',
        '38 -> 60',
        '60++',
    ]

    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, *_
        in detections
    ]

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)
    
    return annotated_image, labels

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Remember to replace the model path with whatever your filepath to the model is

model_path = "/Users/anusha/Desktop/unibots-mp-demo/efficientdet_lite0.tflite"
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path= model_path),
    max_results=5,
    running_mode=VisionRunningMode.VIDEO)

cap = cv2.VideoCapture(0)

with ObjectDetector.create_from_options(options) as detector:
    timestamp_ms = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap in MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Run detection
        result = detector.detect_for_video(mp_image, timestamp_ms)

        # Draw detections
        if result.detections:
            for detection in result.detections:
                bbox = detection.bounding_box
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                label = detection.categories[0].category_name
                score = detection.categories[0].score

                cv2.putText(
                    frame,
                    f"{label} {score:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("MediaPipe Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

        timestamp_ms += 33  # ~30 FPS

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python.vision import FaceLandmarksConnections

MODEL_PATH = "vision/models/face_landmarker.task"
IMAGE_PATH = "test_photo.jpeg"


def detect_face_landmarks(image_path):
    """Detect facial landmarks using MediaPipe."""
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )

    image = mp.Image.create_from_file(image_path)

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(image)

    return result


if __name__ == "__main__":
    result = detect_face_landmarks(IMAGE_PATH)

    if not result.face_landmarks:
        print("No face detected.")
    else:
        landmarks = result.face_landmarks[0]
        print(f"Detected {len(landmarks)} facial landmarks!")

        image = cv2.imread(IMAGE_PATH)
        height, width = image.shape[:2]

        canvas = np.full((height, width, 3), 255, dtype=np.uint8)

        connections = (
            FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
            + FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE
            + FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE
            + FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW
            + FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW
            + FaceLandmarksConnections.FACE_LANDMARKS_LIPS
            + FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS
            + FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS
        )

        for connection in connections:
            start = landmarks[connection.start]
            end = landmarks[connection.end]

            p1 = (int(start.x * width), int(start.y * height))
            p2 = (int(end.x * width), int(end.y * height))

            cv2.line(canvas, p1, p2, (0, 0, 0), 2)


        nose_paths = [
            [168, 6, 197, 195, 5, 4],
            [129, 98, 2, 327, 358],
        ]

        for path in nose_paths:
            for i in range(len(path) - 1):
                start = landmarks[path[i]]
                end = landmarks[path[i + 1]]

                p1 = (int(start.x * width), int(start.y * height))
                p2 = (int(end.x * width), int(end.y * height))

                cv2.line(canvas, p1, p2, (0, 0, 0), 2)


        scale = min(800 / width, 600 / height, 1.0)
        display = cv2.resize(
            canvas, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow("MediaPipe Face Landmarks", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

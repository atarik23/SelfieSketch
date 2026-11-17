import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python.vision import FaceLandmarksConnections
from vision.image_processing import (
    extract_edges,
    image_to_paths,
    thin_lines,
    skeleton_to_paths,
)

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

        edges = extract_edges(IMAGE_PATH)

        # Resize edges to match the original image dimensions
        edges = cv2.resize(edges, (width, height))
        
        # Create a mask around MediaPipe facial features
        mask = np.zeros((height, width), dtype=np.uint8)

        feature_connections = (
            FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE
            + FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE
            + FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW
            + FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW
            + FaceLandmarksConnections.FACE_LANDMARKS_LIPS
        )

        for connection in feature_connections:
            start = landmarks[connection.start]
            end = landmarks[connection.end]

            p1 = (int(start.x * width), int(start.y * height))
            p2 = (int(end.x * width), int(end.y * height))

            cv2.line(mask, p1, p2, 255, 12)

        # Remove Canny edges near these features
        edges[mask > 0] = 0

        # Draw remaining Canny edges
        canvas[edges > 0] = (0, 0, 0) 

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

        skeleton = thin_lines(canvas)

        # Analyze connected components of the skeleton
        binary = (skeleton == 0).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        components = num_labels - 1

        sizes = stats[1:, cv2.CC_STAT_AREA]

        print(f"Skeleton components: {components}")
        print(f"Components with >= 10 pixels: {np.sum(sizes >= 10)}")
        print(f"Largest component: {sizes.max() if len(sizes) else 0} pixels")

        skeleton_paths = skeleton_to_paths(skeleton)

        print(f"Skeleton paths: {len(skeleton_paths)}")
        print(f"Skeleton path points: {sum(len(p) for p in skeleton_paths)}")

        scale = min(800 / width, 600 / height, 1.0)

        cv2.imshow(
            "Skeletonized Drawing",
            cv2.resize(skeleton, None, fx=scale, fy=scale)
        )


        # Convert the final drawing into vector paths
        paths = image_to_paths(canvas)

        print(f"Extracted {len(paths)} paths")
        print(f"Total points: {sum(len(path) for path in paths)}")

        # Visualize extracted paths
        path_preview = np.full_like(canvas, 255)

        for path in paths:
            points = np.array(path, dtype=np.int32)
            cv2.polylines(
                path_preview,
                [points],
                isClosed=False,
                color=(0, 0, 0),
                thickness=1
            )

        preview = cv2.resize(
            path_preview,
            None,
            fx=scale if False else min(800 / width, 600 / height, 1.0),
            fy=min(800 / width, 600 / height, 1.0),
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow("Extracted Paths", preview)

        scale = min(800 / width, 600 / height, 1.0)
        display = cv2.resize(
            canvas, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow("MediaPipe Face Landmarks", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

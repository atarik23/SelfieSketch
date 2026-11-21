import cv2
import numpy as np
from skimage.morphology import skeletonize


def thin_lines(image):
    """Reduce black strokes to a one-pixel-wide skeleton."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    foreground = gray < 127

    skeleton = skeletonize(foreground)

    return np.where(skeleton, 0, 255).astype(np.uint8)

def extract_edges(image_path):
    """Extract simplified edges from an image."""
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # Reduce image resolution
    height, width = image.shape[:2]
    scale = min(1.0, 500 / max(height, width))
    image = cv2.resize(image, None, fx=scale, fy=scale)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smooth small details
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Detect stronger edges
    edges = cv2.Canny(blurred, 40, 120)

    return edges



def image_to_paths(image, min_length=20, epsilon=1.5):
    """Convert black lines on a white image into simplified paths."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    paths = []

    for contour in contours:
        if cv2.arcLength(contour, closed=False) < min_length:
            continue

        simplified = cv2.approxPolyDP(
            contour, epsilon, closed=False
        )

        points = [
            tuple(point[0])
            for point in simplified
        ]

        if len(points) >= 2:
            paths.append(points)

    return paths



def skeleton_to_paths(skeleton, min_points=10):
    """Convert a one-pixel skeleton into drawing paths."""
    pixels = set(zip(*np.where(skeleton == 0)))

    neighbors = {}

    for y, x in pixels:
        neighbors[(y, x)] = [
            (y + dy, x + dx)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy != 0 or dx != 0)
            and (y + dy, x + dx) in pixels
        ]

    def edge(a, b):
        return frozenset((a, b))

    visited = set()
    paths = []

    def trace(start, next_pixel):
        path = [start, next_pixel]
        visited.add(edge(start, next_pixel))

        previous, current = start, next_pixel

        while len(neighbors[current]) == 2:
            candidates = [
                p for p in neighbors[current]
                if p != previous
                and edge(current, p) not in visited
            ]

            if not candidates:
                break

            following = candidates[0]
            visited.add(edge(current, following))
            path.append(following)
            previous, current = current, following

        return path

    # Trace paths between endpoints and junctions
    for pixel in sorted(pixels):
        if len(neighbors[pixel]) != 2:
            for neighbor in neighbors[pixel]:
                if edge(pixel, neighbor) not in visited:
                    path = trace(pixel, neighbor)

                    if len(path) >= min_points:
                        paths.append([
                            (x, y) for y, x in path
                        ])

    # Trace remaining closed loops
    for pixel in sorted(pixels):
        for neighbor in neighbors[pixel]:
            if edge(pixel, neighbor) not in visited:
                path = trace(pixel, neighbor)

                if len(path) >= min_points:
                    paths.append([
                        (x, y) for y, x in path
                    ])

    return paths



def merge_paths(paths, max_distance=3.0):
    """Merge paths whose endpoints are close together."""
    merged = [list(path) for path in paths if len(path) >= 2]

    while True:
        best = None
        best_distance = max_distance

        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a = merged[i]
                b = merged[j]

                # Check all four endpoint combinations
                combinations = [
                    (a[-1], b[0], False, False),
                    (a[-1], b[-1], False, True),
                    (a[0], b[0], True, False),
                    (a[0], b[-1], True, True),
                ]

                for p1, p2, reverse_a, reverse_b in combinations:
                    distance = np.hypot(
                        p1[0] - p2[0],
                        p1[1] - p2[1]
                    )

                    if distance <= best_distance:
                        best_distance = distance
                        best = (i, j, reverse_a, reverse_b)

        if best is None:
            break

        i, j, reverse_a, reverse_b = best

        a = merged[i][::-1] if reverse_a else merged[i]
        b = merged[j][::-1] if reverse_b else merged[j]

        # Avoid duplicating identical endpoints
        if a[-1] == b[0]:
            combined = a + b[1:]
        else:
            combined = a + b

        merged[i] = combined
        merged.pop(j)

    return merged


def simplify_paths(paths, epsilon=1.5):
    """Reduce path points while preserving their overall shape."""
    simplified_paths = []

    for path in paths:
        points = np.array(path, dtype=np.float32).reshape(-1, 1, 2)

        simplified = cv2.approxPolyDP(
            points,
            epsilon,
            closed=False
        )

        if len(simplified) >= 2:
            simplified_paths.append(
                [tuple(point) for point in simplified.reshape(-1, 2)]
            )

    return simplified_paths


if __name__ == "__main__":
    edges = extract_edges("test_photo.jpeg")

    height, width = edges.shape

    scale = min(800 / width, 600 / height, 1.0)

    display = cv2.resize(
        edges,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow("Detected edges", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

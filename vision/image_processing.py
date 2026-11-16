import cv2



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

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    from pathlib import Path
    import cv2, numpy as np
    return Path, cv2


@app.cell
def _(Path):
    # Experimenting with path patterns
    notebook_dir = Path(__file__).parent
    assert notebook_dir == Path("/Users/josephmollo/portfolio/face-access-control/notebooks")
    return


@app.cell
def _(Path):
    # Settle on execution origin relative (root)
    data_dir = Path.cwd() / "data"
    data_dir.mkdir(exist_ok=True)

    # Test images stored in data directory from root
    image_path = data_dir / "face.jpg"
    return (image_path,)


@app.cell
def _(cv2, image_path):
    image = cv2.imread(str(image_path)) # Load image using cv2, must cast path object to string
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Create a greyscale copy
    return gray, image


@app.cell
def _(image):
    image_with_faces = image.copy()
    return (image_with_faces,)


@app.cell
def _(Path, cv2):
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    assert face_cascade is not None
    return (face_cascade,)


@app.cell
def _(face_cascade, gray):
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,      # How much to scale down each iteration
        minNeighbors=5,        # How many neighbors each candidate rectangle should have
        minSize=(30, 30)      # Minimum face size
    )

    return (faces,)


@app.cell
def _(faces):
    faces
    return


@app.cell
def _(cv2, faces, image_with_faces):
    for (x, y, w, h) in faces:
        # Draw rectangle (BGR color format)
        cv2.rectangle(image_with_faces, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        # Optional: Add label
        cv2.putText(
            image_with_faces,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
    return


@app.cell
def _(cv2, image_with_faces):
    from PIL import Image
    image_rgb = cv2.cvtColor(image_with_faces, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    return (pil_image,)


@app.cell
def _(pil_image):
    pil_image
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

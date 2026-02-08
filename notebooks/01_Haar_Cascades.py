import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(rf"""
    #Exploration: Running Haar Cascades on Greyscale Faces
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import cv2, numpy as np
    from PIL import Image
    return Image, Path, cv2, np


@app.cell
def _(Path):
    data_dir: Path = Path.cwd() / "data"
    data_dir.mkdir(exist_ok=True)

    image_path: Path = data_dir / "face.jpg"
    return data_dir, image_path


@app.cell
def _(Path, cv2, image_path: "Path", np):
    image:np.ndarray = cv2.imread(str(image_path)) # im_read: path:str -> np.ndarray
    image_copy = image.copy()
    greyscale:np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    # cvtColor: np.ndarray, oneof(cv2.COLORCONVERSIONCODES) -> np.ndarray 
    #To see full python bindings reference the module stub @
    stub = Path(cv2.__file__).parent / "__init__.pyi"


    assert np.array_equal(image,image_copy)
    """Create greyscale version 'cv2.BGR2GRAY' of image array"""
    return greyscale, image, image_copy


@app.cell
def _(greyscale: "np.ndarray", image: "np.ndarray"):
    assert image.shape[:2] == greyscale.shape[:2] 
    """Validate that greyscale version has same dimensions"""
    return


@app.cell
def _(Path, cv2):
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    assert face_cascade is not None
    """Load classifier into 'face_cascade'"""
    return (face_cascade,)


@app.cell
def _(face_cascade, greyscale: "np.ndarray"):
    faces = face_cascade.detectMultiScale(
        greyscale,
        scaleFactor=1.1,  # How much to scale down each iteration
        minNeighbors=5,  # How many neighbors each candidate rectangle should have
        minSize=(30, 30),  # Minimum face size
    )
    """Detect faces of any scale within the array, return _typing.Sequence[cv2.typing.Rect]"""
    return (faces,)


@app.cell
def _(faces):
    print(type(faces), faces, sep="\n")
    """As can be observed, an array of rectanges is returned"""
    return


@app.cell
def _(cv2, faces, image: "np.ndarray", image_copy):
    for x, y, w, h in faces:
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image_copy,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
    return


@app.cell
def _(Image, image_copy):
    pil_image = Image.fromarray(image_copy)
    pil_image
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Results

    For a single image, of a woman with square centered face and no lack of clarity, a box is generated just below the lips and above the forehead, incorporating both ears. Neither the chin nor the hairline are appreciably present, but approximately half of the jawline is.

    ### Thoughts on greyscale
    Greyscale will effectively map all pixels to the unit interval, representing brightness. This map is much simpler to understand because it would not rely on multivariate periodicity. As much of the basis for signal processing is in frequency domain transforms, and in this case, using haar wavelets, it is simpler to map your domain, if applicable, to something which can be understood in the simplest domain.

    Indeed, this initial model of face classification "Rapid Object Detection using a Boosted Cascade of Simple Features" published in 2001.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Doggo - a misleading pupper
    """)
    return


@app.cell
def _(data_dir: "Path"):
    dog_image_path = data_dir / "dog.jpg"
    return (dog_image_path,)


@app.cell
def _(cv2, dog_image_path):
    dog_image = cv2.imread(
        str(dog_image_path) 
    )  
    dog_gray = cv2.cvtColor(dog_image, cv2.COLOR_BGR2GRAY)  
    return dog_gray, dog_image


@app.cell
def _(dog_gray):
    dog_gray # the numpy representation of the greyscale dog
    return


@app.cell
def _(dog_gray, face_cascade):
    mugs = face_cascade.detectMultiScale(
        dog_gray,
        scaleFactor=1.1,  # How much to scale down each iteration
        minNeighbors=5,  # How many neighbors each candidate rectangle should have
        minSize=(30, 30),  # Minimum face size
    )
    """Detect faces of any scale within the array, return _typing.Sequence[cv2.typing.Rect]"""
    return (mugs,)


@app.cell
def _(cv2, dog_image, mugs):

    def _():
        for x, y, w, h in mugs:
            cv2.rectangle(dog_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                dog_image,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
    _()
    return


@app.cell
def _(cv2, dog_image):
    colorized = cv2.cvtColor(dog_image,code=cv2.COLOR_BGR2RGB) # Not running this leads to a really cool yet kind of weird image with teh red and blue color values swapped
    return (colorized,)


@app.cell
def _(Image, colorized):
    dog_pil_image = Image.fromarray(colorized)
    dog_pil_image
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results
    As is clear, the standard human face classifier as implemented in CV2 does not operate well on canine friends. If one were to clearly inspect the variety of Haar classifiers made available they would see versoins for full profiles of people, turned faces, and cats. This impresses upon us that this technique is suitable for identifying faces when we have a clear frame in which direct pov is provided. Additionally, profiles are very sensitive.
    """)
    return


if __name__ == "__main__":
    app.run()

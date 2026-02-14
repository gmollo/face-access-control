import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import pathlib
    import marimo as mo
    import insightface
    import numpy as np
    from PIL import Image
    import cv2
    return cv2, insightface, mo, np, pathlib


@app.cell
def _(insightface):
    app = insightface.app.FaceAnalysis(
        name = "buffalo_l",
        providers= ['CPUExecutionProvider']
    )
    app.prepare(ctx_id=-1, det_size=(640, 640)) # CTX id specifies GPU/CPU run config, -1 for now since no dedicated GPU 
    """
    the output of this cell represents a full pipeline according to the insightface spec. Seeral different models are used with specialized purposes.
    """
    return (app,)


@app.cell
def _(mo):
    mo.md(r"""
    As seen here:
    face-access-control/.venv/lib/python3.13/site-packages/insightface/app/face_analysis.py, initializing the class amounts to populating a models dict
    """)
    return


@app.cell
def _(path):
    print(path.cwd())
    return


@app.cell
def _(cv2, np, path, pathlib):
    def load_image(p: pathlib.Path)->np.ndarray:
        """
        a simlpe wrapper utility for CV2.imread with input as pathlib Path
        """
        image = cv2.imread(str(path))
        return image
    return (load_image,)


@app.cell
def _(insightface, np):
    def extract_embedding(app: insightface.app.FaceAnalysis, image:np.ndarray):
        """
        app: insightface.app.FaceAnalysis
        image: an RGB encoded numpy array for an image
        """
        faces = app.get(image)

        embeddings = []
        bboxes = [] 

        for face in faces:
            embeddings.append(face.normed_embedding)
            bboxes.append(face.bbox)
        return embeddings, bboxes
    return (extract_embedding,)


@app.cell
def _(pathlib):
    path = pathlib.Path.cwd() / "data" / "raw"/ "RGB_girl.jpg"
    return (path,)


@app.cell
def _(app, extract_embedding, load_image, path):
    image = load_image(path)
    embeddings, bboxes = extract_embedding(app=app,image=image)
    return (embeddings,)


@app.cell
def _(embeddings, np):
    assert len(embeddings[0]) == 512
    assert abs(np.linalg.norm(embeddings[0])-1) < 1e-6
    return


@app.cell
def _(np):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    def visualize_embeddings(embeddings, labels=["girl"]):
        """Visualize embeddings in 2D using t-SNE."""
        # Convert list to numpy array
        embeddings_array = np.array(embeddings)
        n_samples = embeddings_array.shape[0]

        # Adjust perplexity for small sample sizes
        perplexity = min(30, max(5, n_samples - 1))

        # Reduce to 2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings_array)

        # Plot
        plt.figure(figsize=(10, 8))
        for i, label in enumerate(set(labels)):
            mask = [l == label for l in labels]
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=label)
        plt.legend()
        plt.title("Face Embeddings (t-SNE)")
        plt.show()
    return (visualize_embeddings,)


@app.cell
def _(embeddings, visualize_embeddings):
    visualize_embeddings(embeddings, labels=["girl"])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

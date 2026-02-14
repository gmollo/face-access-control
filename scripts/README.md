# Scripts Directory

## download_test_faces.py

Downloads and organizes test face images from the LFW (Labeled Faces in the Wild) dataset.

### Usage

```bash
cd /Users/josephmollo/portfolio/face-access-control
uv run python scripts/download_test_faces.py
```

### What it does

1. Downloads LFW dataset (~170MB)
2. Extracts the tar.gz file
3. Organizes images by person into `data/test/person_XXX/`
4. Selects people with 3+ images
5. Creates up to 50 person folders with multiple images each

### Requirements

- Internet connection
- ~200MB disk space
- Python with `urllib` (standard library)

### Output

```
data/test/
├── person_001/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── person_002/
│   └── ...
└── ...
```

### Notes

- LFW is a public research dataset
- Images are organized by person name
- Script selects people with multiple images for better testing
- Total images: 100+ (depending on available people)

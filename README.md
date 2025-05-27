# Face Finder

A Python application for detecting and matching faces in images using YOLOv8 and InsightFace.

## Features

- Face detection using YOLOv8
- Face encoding and matching using InsightFace
- Support for both local and Google Drive image sources
- Comprehensive logging system
- Organized output structure with timestamped runs

## Requirements

- Python 3.8+
- OpenCV
- YOLOv8
- InsightFace
- Google Drive API (for cloud storage support)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-finder.git
cd face-finder
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For Google Drive support, place your `credentials.json` file in the project root.

## Usage

### Basic Usage

```bash
python face_finder.py --samples /path/to/samples --dataset /path/to/dataset
```

### Command Line Arguments

- `--samples`: Path or URL/ID of the folder containing sample faces
- `--dataset`: Path or URL/ID of the folder containing images to search through
- `--min-detection-confidence`: Minimum detection confidence for face detection (default: 0.5)
- `--local`: Use local folders instead of Google Drive

### Output Structure

The application creates a timestamped output directory for each run:

```
outputs/
└── run_YYYYMMDD_HHMMSS/
    ├── detected_faces/     # Individual face crops
    └── processed_photos/   # Full processed images
        ├── samples/        # Processed sample images
        └── dataset/        # Processed dataset images
```

### Logging

- Logs are stored in the `logs` directory
- Each run creates a new log file with timestamp
- Logs include:
  - Model initialization details
  - Face detection results
  - Face encoding status
  - Comparison results
  - File processing status

## Example

```bash
# Using local folders
python face_finder.py --samples ./samples --dataset ./dataset --local

# Using Google Drive folders
python face_finder.py --samples "https://drive.google.com/drive/folders/your-folder-id" --dataset "https://drive.google.com/drive/folders/your-folder-id"
```

## Output Files

- `detected_faces/`: Contains individual face crops from detected faces
- `processed_photos/`: Contains full images with face detection boxes and match annotations
- `logs/`: Contains detailed logs of each run

## Notes

- The application uses YOLOv8 for face detection
- Face matching is performed using InsightFace's face recognition
- All outputs are organized by run timestamp for easy tracking and comparison
- Logs provide detailed information about the detection and matching process 
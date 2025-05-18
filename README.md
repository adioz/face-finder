# Face Finder

A Python application that detects faces in images from Google Drive folders and identifies matching faces between an example folder and a scan folder.

## Features

- Connects to Google Drive API
- Downloads images from specified Google Drive folders
- Detects faces using YOLOv8s-face (optimized for group and far-sighted photos)
- Extracts face encodings using MediaPipe Face Mesh
- Compares faces between example and scan folders
- Marks detected faces with bounding boxes:
  - Green boxes for all detected faces
  - Red boxes for faces that match the example folder
- Saves processed images to a local directory

## Setup

1. Install dependencies:
```bash
pip install ultralytics mediapipe opencv-python google-api-python-client google-auth-oauthlib python-dotenv
```

2. Set up Google Drive API:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the Google Drive API
   - Create OAuth 2.0 credentials
   - Download the credentials and save them as `credentials.json` in the project directory

3. Run the application:
```bash
python face_finder.py --example-folder "FOLDER_URL_OR_ID" --scan-folder "FOLDER_URL_OR_ID" --min-detection-confidence 0.5
```

For example:
```bash
python face_finder.py --example-folder "https://drive.google.com/drive/folders/177KQJkQk2i1Fo2qzy6QWAetVyZfVCYsN" --scan-folder "https://drive.google.com/drive/folders/1AGLYLm5rmCZYz1OxM4wQMGvUA_culwIb" --min-detection-confidence 0.5
```

The first time you run the application, it will:
1. Download the YOLOv8s-face model (if not already downloaded)
2. Open a browser window for Google OAuth authentication
3. Ask you to log in to your Google account
4. Request permission to access your Google Drive
5. Save the authentication token for future use

## Output

Processed images will be saved in the `processed_photos` directory. Each image will have:
- Green rectangles around all detected faces
- Red rectangles around faces that match the example folder
- Confidence scores displayed above each face
- Files from the example folder will be prefixed with "example_"
- Files from the scan folder will be prefixed with "scan_"

Cropped face images will be saved in the `detected_faces` directory for inspection.

## Notes

- The application uses YOLOv8s-face for face detection, which is optimized for group photos and far-sighted faces
- Face encoding is done using MediaPipe Face Mesh for accurate face matching
- Only image files (jpg, png, etc.) will be processed
- Make sure the Google Drive folders are shared with the Google account you use for authentication
- Face matching is done using a combination of face mesh landmarks and Euclidean distance comparison
- The matching threshold can be adjusted in the code (default is 0.8)
- The minimum detection confidence can be adjusted via command line (default is 0.5)

from ultralytics import YOLO

# Load the pre-trained YOLOv8 face detection model
model = YOLO('yolov8n-face.pt')  # or 'yolov8s-face.pt' or 'yolov8m-face.pt'

# Run inference on an image
results = model('path/to/your/image.jpg')

# Process results
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Box coordinates
        confidence = box.conf[0]  # Confidence score
        print(f"Face detected at {x1}, {y1}, {x2}, {y2} with confidence {confidence}") 
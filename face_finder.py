import os
import cv2
import mediapipe as mp
import numpy as np
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import pickle
from dotenv import load_dotenv
import argparse
import re
from typing import List, Tuple
from ultralytics import YOLO
from pathlib import Path
from pillow_heif import register_heif_opener
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
import sys
from datetime import datetime
import logging

# Register HEIF opener with PIL
register_heif_opener()

# Load environment variables
load_dotenv()

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')
        self.closed = False
        self.write("\n" + "="*80 + "\n")
        self.write("NEW SESSION STARTED\n")
        self.write("="*80 + "\n\n")

    def write(self, message):
        if message.strip() and not self.closed:  # Only log non-empty messages
            # Ensure message ends with a newline
            if not message.endswith('\n'):
                message += '\n'
            self.terminal.write(message)
            try:
                self.log.write(message)
                self.log.flush()
            except ValueError:
                # If file is closed, just write to terminal
                pass

    def flush(self):
        if not self.closed:
            self.terminal.flush()
            try:
                self.log.flush()
            except ValueError:
                pass

    def close(self):
        if not self.closed:
            self.write("\n" + "="*80 + "\n")
            self.write("SESSION ENDED\n")
            self.write("="*80 + "\n\n")
            self.log.close()
            self.closed = True

class FaceFinder:
    def __init__(self, min_detection_confidence=0.5, use_local=False):
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Set up logging
        log_filename = f'logs/face_finder_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        self.logger = TeeLogger(log_filename)
        sys.stdout = self.logger
        
        self.min_detection_confidence = min_detection_confidence
        self.sample_faces = []  # List to store face encodings from sample folder
        self.use_local = use_local

        # Create output directory structure
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path('outputs') / f'run_{self.run_timestamp}'
        self.detected_faces_dir = self.output_dir / 'detected_faces'
        self.processed_photos_dir = self.output_dir / 'processed_photos'
        
        # Create all necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detected_faces_dir.mkdir(exist_ok=True)
        self.processed_photos_dir.mkdir(exist_ok=True)
        
        print(f"\nCreated output directory structure:")
        print(f"  - Main output directory: {self.output_dir}")
        print(f"  - Detected faces directory: {self.detected_faces_dir}")
        print(f"  - Processed photos directory: {self.processed_photos_dir}\n")

        # Initialize YOLOv8 model
        print("Initializing YOLOv8 face detection model...")
        self.face_detector = YOLO('yolov8n-face.pt')  # Use face detection model
        print(f"Model loaded successfully!")
        print(f"Model info: {self.face_detector.model.names}")
        print(f"Model path: {self.face_detector.model.pt_path}\n")

        # Initialize InsightFace
        print("Initializing InsightFace model...")
        self.face_analyzer = FaceAnalysis(name='buffalo_l', root='.')
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        print("InsightFace model loaded successfully!")
        print("Model details:")
        print(f"  - Model name: buffalo_l")
        print(f"  - Detection size: 640x640")
        print(f"  - Available models: {list(self.face_analyzer.models.keys())}\n")

        self.creds = None
        self.service = None
        if not use_local:
            self.authenticate()

    def __del__(self):
        # Close the logger when the object is destroyed
        if hasattr(self, 'logger'):
            self.logger.close()
            sys.stdout = self.logger.terminal

    def authenticate(self):
        """Authenticate with Google Drive API."""
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                self.creds = pickle.load(token)
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                self.creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(self.creds, token)
        self.service = build('drive', 'v3', credentials=self.creds)

    def extract_folder_id(self, folder_url):
        """Extract folder ID from Google Drive URL."""
        match = re.search(r'/folders/([a-zA-Z0-9_-]+)', folder_url)
        if match:
            return match.group(1)
        return folder_url  # Assume it's already a folder ID

    def list_files_in_folder(self, folder_path):
        """List all image files in a folder (local or Google Drive)."""
        if self.use_local:
            folder_path = Path(folder_path)
            if not folder_path.exists():
                raise ValueError(f"Local folder not found: {folder_path}")
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.heic', '.HEIC'}
            files = []
            for file_path in folder_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    files.append({
                        'id': str(file_path),
                        'name': file_path.name,
                        'mimeType': f'image/{file_path.suffix[1:]}'
                    })
            print(f"Found {len(files)} images in {folder_path}")
            return files
        else:
            query = f"'{folder_path}' in parents and mimeType contains 'image/'"
            results = self.service.files().list(
                q=query,
                pageSize=1000,
                fields="files(id, name, mimeType)"
            ).execute()
            return results.get('files', [])

    def download_file(self, file_id):
        """Download a file from Google Drive or read from local path."""
        if self.use_local:
            file_path = Path(file_id)
            if file_path.suffix.lower() in {'.heic', '.HEIC'}:
                # Handle HEIC files using PIL
                image = Image.open(file_path)
                # Convert to numpy array
                image = np.array(image)
                # Convert to BGR for OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # Convert to bytes
                _, buffer = cv2.imencode('.jpg', image)
                return io.BytesIO(buffer)
            else:
                with open(file_id, 'rb') as f:
                    return io.BytesIO(f.read())
        else:
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)
            return fh

    def get_face_encoding(self, image: np.ndarray, face_location: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract face encoding using InsightFace."""
        x, y, w, h = face_location
        
        # Expand the bounding box by 20%
        margin = 0.2
        x_new = max(0, x - int(w * margin))
        y_new = max(0, y - int(h * margin))
        w_new = min(image.shape[1] - x_new, w + int(w * margin * 2))
        h_new = min(image.shape[0] - y_new, h + int(h * margin * 2))
        face_crop = image[y_new:y_new+h_new, x_new:x_new+w_new]
        
        # Convert to RGB for InsightFace
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        try:
            # Get face embedding using InsightFace
            faces = self.face_analyzer.get(face_crop)
            if faces:
                print(f"    Successfully encoded face: {w}x{h} pixels (expanded to {w_new}x{h_new})")
                return faces[0].embedding
            else:
                print(f"    No face detected in cropped region: {w_new}x{h_new} pixels")
                return None
        except Exception as e:
            print(f"    Error during face encoding: {str(e)}")
            return None

    def compare_faces(self, face_encoding1: np.ndarray, face_encoding2: np.ndarray, threshold: float = 0.5, 
                     scan_face_idx: int = None, scan_image: str = None, 
                     sample_image: str = None) -> Tuple[bool, float]:
        """Compare two face encodings using cosine similarity."""
        if face_encoding1 is None or face_encoding2 is None:
            return False, 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(face_encoding1, face_encoding2) / (np.linalg.norm(face_encoding1) * np.linalg.norm(face_encoding2))
        
        # Print detailed comparison info with context
        context = f"Scan face {scan_face_idx} in {scan_image} vs Sample from {sample_image}"
        print(f"{context} - Similarity: {similarity:.4f}, Threshold: {threshold}")
        
        return similarity > threshold, similarity

    def detect_and_encode_faces(self, image_data, prefix="") -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], List[np.ndarray]]:
        try:
            print(f"\nProcessing image: {prefix}")
            print("-" * 50)
            
            nparr = np.frombuffer(image_data.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Error: Could not decode image for {prefix}")
                return None, [], []
            
            print(f"Successfully decoded image with shape {image.shape}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = []
            face_encodings = []
            
            # Run YOLOv8 face detection
            print("\nRunning YOLOv8 face detection...")
            results = self.face_detector(image_rgb)
            
            if results[0].boxes:
                print(f"Found {len(results[0].boxes)} face(s) in the image")
                for idx, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    if confidence >= self.min_detection_confidence:
                        width = x2 - x1
                        height = y2 - y1
                        face_location = (x1, y1, width, height)
                        print(f"\nFace {idx+1}:")
                        print(f"  - Location: (x={x1}, y={y1}, w={width}, h={height})")
                        print(f"  - Confidence: {confidence:.2f}")
                        
                        # Save cropped face for inspection
                        face_crop = image[max(y1,0):max(y1,0)+height, max(x1,0):max(x1,0)+width]
                        if face_crop.size > 0:
                            crop_path = os.path.join(self.detected_faces_dir, f"{prefix}_face_{idx+1}.jpg")
                            cv2.imwrite(crop_path, face_crop)
                            print(f"  - Cropped face saved to: {crop_path}")
                        
                        face_locations.append(face_location)
                        
                        # Get face encoding using InsightFace
                        face_encoding = self.get_face_encoding(image, face_location)
                        if face_encoding is not None:
                            face_encodings.append(face_encoding)
                            print("  - Face encoding successful")
                        else:
                            print("  - Face encoding failed")
                        
                        # Draw rectangle
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                print("No faces found in the image")
                    
            return image, face_locations, face_encodings
        except Exception as e:
            print(f"Error processing image {prefix}: {str(e)}")
            return None, [], []
    
    def process_folders(self, sample_folder_path, scan_folder_path):
        """Process images from folders and find matching faces."""
        if not self.service and not self.use_local:
            self.authenticate()
            
        # Create subdirectories for samples and dataset
        samples_output_dir = self.processed_photos_dir / 'samples'
        dataset_output_dir = self.processed_photos_dir / 'dataset'
        samples_output_dir.mkdir(exist_ok=True)
        dataset_output_dir.mkdir(exist_ok=True)
        print(f"Created/verified output directories:")
        print(f"  - Samples: {samples_output_dir.absolute()}")
        print(f"  - Dataset: {dataset_output_dir.absolute()}")
        
        # Extract folder IDs
        sample_folder_id = self.extract_folder_id(sample_folder_path)
        scan_folder_id = self.extract_folder_id(scan_folder_path)
        
        # Get files from both folders
        sample_files = self.list_files_in_folder(sample_folder_id)
        scan_files = self.list_files_in_folder(scan_folder_id)
        
        print(f"Found {len(sample_files)} images in sample folder")
        print(f"Found {len(scan_files)} images in scan folder")
        
        # Process sample folder images and collect face encodings
        print("\nProcessing sample folder images:")
        for i, file in enumerate(sample_files):
            print(f"\nProcessing sample image {i+1}/{len(sample_files)}: {file['name']}")
            try:
                image_data = self.download_file(file['id'])
                prefix = f"sample_{file['name']}"
                image, face_locations, face_encodings = self.detect_and_encode_faces(image_data, prefix=prefix)
                
                if image is not None:
                    # Save processed image
                    output_path = samples_output_dir / f'{prefix}.jpg'
                    print(f"Attempting to save image to {output_path}")
                    success = cv2.imwrite(str(output_path), image)
                    if success:
                        print(f"Successfully saved processed image to {output_path}")
                    else:
                        print(f"Failed to save image to {output_path}")
                    
                    if face_encodings:
                        # Store face encodings with their source image name
                        for encoding in face_encodings:
                            self.sample_faces.append((encoding, file['name']))
                        print(f"Added {len(face_encodings)} face encodings from this image")
                else:
                    print(f"Failed to process image {file['name']}")
            except Exception as e:
                print(f"Error processing sample image {file['name']}: {str(e)}")
        
        print(f"\nCollected {len(self.sample_faces)} face encodings from sample folder")
        
        # Process scan folder images and find matches
        print("\nProcessing scan folder images:")
        matches_found = False
        for i, file in enumerate(scan_files):
            print(f"\nProcessing scan image {i+1}/{len(scan_files)}: {file['name']}")
            try:
                image_data = self.download_file(file['id'])
                prefix = f"scan_{file['name']}"
                image, face_locations, face_encodings = self.detect_and_encode_faces(image_data, prefix=prefix)
                
                if image is not None:
                    # Check for matches with sample faces
                    for j, scan_encoding in enumerate(face_encodings):
                        best_match = None
                        best_similarity = 0
                        
                        for k, (sample_encoding, sample_name) in enumerate(self.sample_faces):
                            is_match, similarity = self.compare_faces(
                                scan_encoding, sample_encoding,
                                scan_face_idx=j+1,
                                scan_image=file['name'],
                                sample_image=sample_name
                            )
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = (k, sample_name)
                        
                        if best_match and best_similarity > 0.5:  # Lowered threshold for matching
                            k, sample_name = best_match
                            # Draw red rectangle for matching faces
                            x, y, w, h = face_locations[j]
                            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(image, f"Match: {sample_name} ({best_similarity:.2f})", 
                                      (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            print(f"Match found: Scan face {j+1} in {file['name']} matches Sample face from {sample_name} (similarity: {best_similarity:.4f})")
                            matches_found = True
                    
                    # Save processed image
                    output_path = dataset_output_dir / f'{prefix}.jpg'
                    print(f"Attempting to save image to {output_path}")
                    success = cv2.imwrite(str(output_path), image)
                    if success:
                        print(f"Successfully saved processed image to {output_path}")
                    else:
                        print(f"Failed to save image to {output_path}")
                else:
                    print(f"Failed to process image {file['name']}")
            except Exception as e:
                print(f"Error processing scan image {file['name']}: {str(e)}")
        
        if matches_found:
            print("\nFound matching faces in scan folder images!")
        else:
            print("\nNo matching faces found in scan folder images.")

def main():
    parser = argparse.ArgumentParser(description='Process folders for face detection and matching.')
    parser.add_argument('--samples', required=True, help='Path or URL/ID of the folder containing sample faces')
    parser.add_argument('--dataset', required=True, help='Path or URL/ID of the folder containing images to search through')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5, help='Minimum detection confidence for face detection (default: 0.5)')
    parser.add_argument('--local', action='store_true', help='Use local folders instead of Google Drive')
    args = parser.parse_args()
    
    face_finder = FaceFinder(min_detection_confidence=args.min_detection_confidence, use_local=args.local)
    face_finder.process_folders(args.samples, args.dataset)

if __name__ == "__main__":
    main() 
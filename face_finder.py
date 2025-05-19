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

# Load environment variables
load_dotenv()

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class FaceFinder:
    def __init__(self, min_detection_confidence=0.7):
        self.min_detection_confidence = min_detection_confidence
        self.example_faces = []  # List to store face encodings from example folder
        self.detected_faces_dir = 'detected_faces'
        os.makedirs(self.detected_faces_dir, exist_ok=True)

        # Initialize YOLOv8 model
        print("Loading YOLOv8 model...")
        self.face_detector = YOLO('yolov8x.pt')  # Use the largest model for best accuracy
        print("Model loaded successfully!")

        # Initialize MediaPipe Face Mesh for encoding
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )

        self.creds = None
        self.service = None

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
        
    def list_files_in_folder(self, folder_id):
        """List all image files in a Google Drive folder."""
        query = f"'{folder_id}' in parents and mimeType contains 'image/'"
        results = self.service.files().list(
            q=query,
            pageSize=1000,
            fields="files(id, name, mimeType)"
        ).execute()
        return results.get('files', [])
    
    def download_file(self, file_id):
        """Download a file from Google Drive."""
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        return fh
    
    def get_face_encoding(self, image: np.ndarray, face_location: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract face encoding using MediaPipe Face Mesh with improved cropping."""
        x, y, w, h = face_location
        # Expand the bounding box by 20%
        margin = 0.2
        x_new = max(0, x - int(w * margin))
        y_new = max(0, y - int(h * margin))
        w_new = min(image.shape[1] - x_new, w + int(w * margin * 2))
        h_new = min(image.shape[0] - y_new, h + int(h * margin * 2))
        face_crop = image[y_new:y_new+h_new, x_new:x_new+w_new]
        
        # Resize to a consistent aspect ratio (1:1)
        face_crop = cv2.resize(face_crop, (224, 224))
        
        # Convert to RGB for MediaPipe
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(face_crop)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            # Convert landmarks to a flat array of coordinates
            encoding = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            return encoding
        return None
    
    def compare_faces(self, face_encoding1: np.ndarray, face_encoding2: np.ndarray, tolerance: float = 0.7) -> bool:
        """Compare two face encodings using cosine similarity and return True if they match."""
        if face_encoding1 is None or face_encoding2 is None:
            return False
        # Normalize the encodings
        face_encoding1_norm = face_encoding1 / np.linalg.norm(face_encoding1)
        face_encoding2_norm = face_encoding2 / np.linalg.norm(face_encoding2)
        # Calculate cosine similarity
        similarity = np.dot(face_encoding1_norm, face_encoding2_norm)
        print(f"Cosine similarity: {similarity:.4f}, Threshold: {tolerance}")
        return similarity > tolerance
    
    def detect_and_encode_faces(self, image_data, prefix="") -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], List[np.ndarray]]:
        nparr = np.frombuffer(image_data.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            print("Error: Could not decode image")
            return None, [], []
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = []
        face_encodings = []
        
        # Run YOLOv8 detection
        results = self.face_detector(image_rgb, classes=[0])  # class 0 is person in COCO dataset
        
        if results[0].boxes:
            print(f"Found {len(results[0].boxes)} face(s) in the image")
            for idx, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                if confidence >= self.min_detection_confidence:
                    width = x2 - x1
                    height = y2 - y1
                    face_location = (x1, y1, width, height)
                    print(f"  Face {idx+1}: (x={x1}, y={y1}, w={width}, h={height}), confidence={confidence:.2f}")
                    
                    # Save cropped face for inspection
                    face_crop = image[max(y1,0):max(y1,0)+height, max(x1,0):max(x1,0)+width]
                    if face_crop.size > 0:
                        crop_path = os.path.join(self.detected_faces_dir, f"{prefix}_face_{idx+1}.jpg")
                        cv2.imwrite(crop_path, face_crop)
                        print(f"    Cropped face saved to {crop_path}")
                    
                    face_locations.append(face_location)
                    
                    # Get face encoding using MediaPipe Face Mesh
                    face_encoding = self.get_face_encoding(image, face_location)
                    if face_encoding is not None:
                        face_encodings.append(face_encoding)
                    
                    # Draw rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("No faces found in the image")
                
        return image, face_locations, face_encodings
    
    def process_folders(self, example_folder_url, scan_folder_url):
        """Process images from Google Drive folders and find matching faces."""
        if not self.service:
            self.authenticate()
            
        # Create output directory if it doesn't exist
        os.makedirs('processed_photos', exist_ok=True)
        
        # Extract folder IDs
        example_folder_id = self.extract_folder_id(example_folder_url)
        scan_folder_id = self.extract_folder_id(scan_folder_url)
        
        # Get files from both folders
        example_files = self.list_files_in_folder(example_folder_id)
        scan_files = self.list_files_in_folder(scan_folder_id)
        
        print(f"Found {len(example_files)} images in example folder")
        print(f"Found {len(scan_files)} images in scan folder")
        
        # Process example folder images and collect face encodings
        print("\nProcessing example folder images:")
        for i, file in enumerate(example_files):
            print(f"Processing example image {i+1}/{len(example_files)}: {file['name']}")
            image_data = self.download_file(file['id'])
            prefix = f"example_{file['id']}"
            image, face_locations, face_encodings = self.detect_and_encode_faces(image_data, prefix=prefix)
            
            if image is not None and face_encodings:
                # Save processed image
                output_path = os.path.join('processed_photos', f'{prefix}.jpg')
                cv2.imwrite(output_path, image)
                print(f"Saved processed image to {output_path}")
                
                # Store face encodings
                self.example_faces.extend(face_encodings)
        
        print(f"\nCollected {len(self.example_faces)} face encodings from example folder")
        
        # Process scan folder images and find matches
        print("\nProcessing scan folder images:")
        matches_found = False
        for i, file in enumerate(scan_files):
            print(f"Processing scan image {i+1}/{len(scan_files)}: {file['name']}")
            image_data = self.download_file(file['id'])
            prefix = f"scan_{file['id']}"
            image, face_locations, face_encodings = self.detect_and_encode_faces(image_data, prefix=prefix)
            
            if image is not None and face_encodings:
                # Check for matches with example faces
                for j, scan_encoding in enumerate(face_encodings):
                    for k, example_encoding in enumerate(self.example_faces):
                        if self.compare_faces(scan_encoding, example_encoding):
                            # Draw red rectangle for matching faces
                            x, y, w, h = face_locations[j]
                            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(image, f"Match #{k+1}", (x, y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            print(f"Match found: Scan face {j+1} in {file['name']} matches Example face {k+1} in {example_files[k]['name']}")
                            matches_found = True
                
                # Save processed image
                output_path = os.path.join('processed_photos', f'{prefix}.jpg')
                cv2.imwrite(output_path, image)
                print(f"Saved processed image to {output_path}")
        
        if matches_found:
            print("\nFound matching faces in scan folder images!")
        else:
            print("\nNo matching faces found in scan folder images.")

def main():
    parser = argparse.ArgumentParser(description='Process Google Drive folders for face detection and matching.')
    parser.add_argument('--example-folder', required=True, help='URL or ID of the example folder')
    parser.add_argument('--scan-folder', required=True, help='URL or ID of the folder to scan')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5, help='Minimum detection confidence for face detection (default: 0.5)')
    args = parser.parse_args()
    
    face_finder = FaceFinder(min_detection_confidence=args.min_detection_confidence)
    face_finder.process_folders(args.example_folder, args.scan_folder)

if __name__ == "__main__":
    main() 
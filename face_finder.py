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
        self.original_stdout = sys.stdout  # Store the original stdout
        self.terminal = self.original_stdout # Keep a reference for potential direct use if needed
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
            self.original_stdout.write(message) # Write to the original stdout
            try:
                self.log.write(message)
                self.log.flush()
            except ValueError:
                # If file is closed, just write to terminal
                pass

    def flush(self):
        if not self.closed:
            self.original_stdout.flush() # Flush the original stdout
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
            # Restore original stdout when FaceFinder is deleted only if TeeLogger was the one that set it
            if isinstance(sys.stdout, TeeLogger) and sys.stdout == self.logger:
                 sys.stdout = self.logger.original_stdout
            self.logger.close()

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
                # This error is critical for the specific folder, so it's raised.
                # It will be caught in process_folders()
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

    def download_file(self, file_id, file_name_for_error_reporting="Unknown file"):
        """Download a file from Google Drive or read from local path."""
        try:
            if self.use_local:
                file_path = Path(file_id)
                if not file_path.exists():
                    raise FileNotFoundError(f"Local file not found: {file_path}")

                if file_path.suffix.lower() in {'.heic', '.HEIC'}:
                    print(f"Processing HEIC file: {file_path}")
                    image = Image.open(file_path)
                    image = np.array(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    _, buffer = cv2.imencode('.jpg', image)
                    return io.BytesIO(buffer)
                else:
                    with open(file_id, 'rb') as f:
                        return io.BytesIO(f.read())
            else: # Google Drive
                request = self.service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                fh.seek(0)
                return fh
        except FileNotFoundError as e:
            print(f"Error downloading/reading file '{file_name_for_error_reporting}' (ID: {file_id}): {e}")
            raise # Re-raise to be caught by process_folders
        except Exception as e:
            # Catch other potential errors (e.g., PIL errors for corrupted HEIC)
            print(f"Error processing file '{file_name_for_error_reporting}' (ID: {file_id}) during download/conversion: {e}")
            raise # Re-raise to be caught by process_folders

    def get_face_encoding(self, image: np.ndarray, face_location: Tuple[int, int, int, int], image_name: str = "Unknown image") -> np.ndarray:
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
                print(f"    Successfully encoded face: {w}x{h} pixels (expanded to {w_new}x{h_new}) from {image_name}")
                return faces[0].embedding
            else:
                print(f"    No face detected by InsightFace in cropped region: {w_new}x{h_new} pixels from {image_name}")
                return None
        except Exception as e:
            print(f"    Error during InsightFace encoding for {image_name}: {str(e)}")
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
        context = f"Scan face #{scan_face_idx} in '{scan_image}' vs Sample from '{sample_image}'"
        print(f"{context} - Similarity: {similarity:.4f}, Threshold: {threshold}")
        
        return similarity > threshold, similarity

    def detect_and_encode_faces(self, image_data, image_name_prefix="") -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], List[np.ndarray]]:
        try:
            print(f"\nProcessing image: {image_name_prefix}")
            print("-" * 50)
            
            nparr = np.frombuffer(image_data.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Error: Could not decode image: {image_name_prefix}")
                return None, [], []
            
            print(f"Successfully decoded {image_name_prefix} with shape {image.shape}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = []
            face_encodings = []
            
            # Run YOLOv8 face detection
            print(f"\nRunning YOLOv8 face detection on {image_name_prefix}...")
            results = self.face_detector(image_rgb)
            
            if results[0].boxes:
                print(f"Found {len(results[0].boxes)} potential face(s) in {image_name_prefix}")
                for idx, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    if confidence >= self.min_detection_confidence:
                        width = x2 - x1
                        height = y2 - y1
                        face_location = (x1, y1, width, height)
                        print(f"\nFace #{idx+1} in {image_name_prefix}:")
                        print(f"  - Location: (x={x1}, y={y1}, w={width}, h={height})")
                        print(f"  - Confidence: {confidence:.2f}")
                        
                        # Save cropped face for inspection
                        face_crop = image[max(y1,0):max(y1,0)+height, max(x1,0):max(x1,0)+width]
                        if face_crop.size > 0:
                            crop_filename = f"{image_name_prefix}_face_{idx+1}.jpg"
                            # Sanitize crop_filename by replacing slashes or colons if image_name_prefix contains them
                            crop_filename = crop_filename.replace('/', '_').replace('\\', '_').replace(':', '_')
                            crop_path = self.detected_faces_dir / crop_filename
                            try:
                                cv2.imwrite(str(crop_path), face_crop)
                                print(f"  - Cropped face saved to: {crop_path}")
                            except Exception as e_write:
                                print(f"  - Failed to save cropped face to {crop_path}: {e_write}")

                        face_locations.append(face_location)
                        
                        # Get face encoding using InsightFace
                        face_encoding = self.get_face_encoding(image, face_location, image_name=image_name_prefix)
                        if face_encoding is not None:
                            face_encodings.append(face_encoding)
                            print(f"  - Face encoding successful for face #{idx+1} in {image_name_prefix}")
                        else:
                            print(f"  - Face encoding failed for face #{idx+1} in {image_name_prefix}")
                        
                        # Draw rectangle
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, f"Conf: {confidence:.2f}", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                print(f"No faces detected by YOLOv8 in {image_name_prefix}")
                    
            return image, face_locations, face_encodings
        except Exception as e:
            print(f"Error processing image {image_name_prefix}: {str(e)}")
            return None, [], []
    
    def process_folders(self, sample_folder_path, scan_folder_path):
        """Process images from folders and find matching faces."""
        if not self.service and not self.use_local:
            try:
                self.authenticate()
            except Exception as e:
                print(f"Error: Google Drive authentication failed: {e}")
                # If authentication fails and we are not using local, we cannot proceed.
                # This error will be caught by the GUI's main try-except.
                raise RuntimeError(f"Google Drive authentication failed: {e}")

        # Create subdirectories for samples and dataset
        samples_output_dir = self.processed_photos_dir / 'samples'
        dataset_output_dir = self.processed_photos_dir / 'dataset'
        samples_output_dir.mkdir(exist_ok=True)
        dataset_output_dir.mkdir(exist_ok=True)
        print(f"Created/verified output directories:")
        print(f"  - Samples output: {samples_output_dir.resolve()}")
        print(f"  - Dataset output: {dataset_output_dir.resolve()}")
        
        # Extract folder IDs
        sample_folder_id = self.extract_folder_id(sample_folder_path) if not self.use_local else sample_folder_path
        scan_folder_id = self.extract_folder_id(scan_folder_path) if not self.use_local else scan_folder_path
        
        sample_files = []
        scan_files = []

        try:
            print(f"\nListing files in sample folder: {sample_folder_id}...")
            sample_files = self.list_files_in_folder(sample_folder_id)
            print(f"Found {len(sample_files)} images in sample folder: '{sample_folder_id}'")
        except ValueError as e:
            print(f"Error listing files in sample folder '{sample_folder_id}': {e}")
            # If sample folder can't be listed, we can't proceed with samples.
            # We might still be able to process the dataset folder if it's just for detection without comparison.
            # However, the current logic relies on sample_faces. So, we stop if sample files are critical.
            # For this application, sample faces are essential.
            raise RuntimeError(f"Could not process sample folder '{sample_folder_id}': {e}") # Critical error

        try:
            print(f"\nListing files in dataset folder: {scan_folder_id}...")
            scan_files = self.list_files_in_folder(scan_folder_id)
            print(f"Found {len(scan_files)} images in dataset folder: '{scan_folder_id}'")
        except ValueError as e:
            print(f"Error listing files in dataset folder '{scan_folder_id}': {e}")
            # If dataset folder can't be listed, we can't proceed with scanning.
            raise RuntimeError(f"Could not process dataset folder '{scan_folder_id}': {e}") # Critical error

        # Process sample folder images and collect face encodings
        print("\n--- Processing Sample Folder Images ---")
        for i, file_info in enumerate(sample_files):
            file_id = file_info['id']
            file_name = file_info['name']
            print(f"\nProcessing sample image {i+1}/{len(sample_files)}: '{file_name}' (ID: {file_id})")
            try:
                image_data = self.download_file(file_id, file_name_for_error_reporting=file_name)
                if image_data is None: # download_file now raises on error, so this check might be redundant
                    print(f"Skipping sample image {file_name} due to download/read error.")
                    continue
                    
                # Sanitize file_name for use as a prefix (e.g., replace slashes if name contains parts of path)
                safe_prefix_name = file_name.replace('/', '_').replace('\\', '_')
                prefix = f"sample_{safe_prefix_name}"

                processed_image, face_locations, face_encodings = self.detect_and_encode_faces(image_data, image_name_prefix=prefix)

                if processed_image is not None:
                    output_filename = f"{prefix}.jpg"
                    output_path = samples_output_dir / output_filename
                    print(f"Attempting to save processed sample image to {output_path}")
                    try:
                        success = cv2.imwrite(str(output_path), processed_image)
                        if success:
                            print(f"Successfully saved processed sample image to {output_path}")
                        else:
                            print(f"Failed to save processed sample image to {output_path} (cv2.imwrite returned false)")
                    except Exception as e_write_img:
                        print(f"Error saving processed sample image {output_filename} to {output_path}: {e_write_img}")

                    if face_encodings:
                        for encoding in face_encodings:
                            self.sample_faces.append((encoding, file_name)) # Use original file_name for reference
                        print(f"Added {len(face_encodings)} face encodings from sample image '{file_name}'")
                else:
                    print(f"Failed to process sample image '{file_name}' (detect_and_encode_faces returned None)")
            except FileNotFoundError: # Already printed in download_file, but good to catch here to continue
                print(f"Skipping sample file '{file_name}' as it was not found or could not be read.")
            except Exception as e:
                print(f"Error processing sample image '{file_name}' (ID: {file_id}): {str(e)}")
        
        if not self.sample_faces:
            print("\nWarning: No face encodings collected from the sample folder. Matching will not be possible.")
        else:
            print(f"\nCollected {len(self.sample_faces)} face encodings from sample folder.")
        
        # Process scan folder images and find matches
        print("\n--- Processing Dataset Folder Images ---")
        matches_found_overall = False
        for i, file_info in enumerate(scan_files):
            file_id = file_info['id']
            file_name = file_info['name']
            print(f"\nProcessing dataset image {i+1}/{len(scan_files)}: '{file_name}' (ID: {file_id})")
            try:
                image_data = self.download_file(file_id, file_name_for_error_reporting=file_name)
                if image_data is None: # download_file now raises on error
                    print(f"Skipping dataset image {file_name} due to download/read error.")
                    continue

                safe_prefix_name = file_name.replace('/', '_').replace('\\', '_')
                prefix = f"dataset_{safe_prefix_name}"
                
                processed_image, face_locations, face_encodings = self.detect_and_encode_faces(image_data, image_name_prefix=prefix)

                if processed_image is not None:
                    image_had_match = False
                    if not self.sample_faces:
                        print(f"  No sample faces loaded, skipping comparison for faces in '{file_name}'.")
                    elif face_encodings:
                        print(f"  Comparing {len(face_encodings)} detected face(s) in '{file_name}' against {len(self.sample_faces)} sample face(s).")
                        for j, scan_encoding in enumerate(face_encodings):
                            best_match_for_this_face = None
                            highest_similarity_for_this_face = 0.0

                            for k, (sample_encoding, sample_name) in enumerate(self.sample_faces):
                                is_match, similarity = self.compare_faces(
                                    scan_encoding, sample_encoding,
                                    scan_face_idx=j + 1,
                                    scan_image=file_name,
                                    sample_image=sample_name
                                )
                                if similarity > highest_similarity_for_this_face:
                                    highest_similarity_for_this_face = similarity
                                    if is_match: # is_match considers the threshold
                                        best_match_for_this_face = (k, sample_name, similarity)

                            if best_match_for_this_face:
                                k_idx, matched_sample_name, match_similarity = best_match_for_this_face
                                x, y, w, h = face_locations[j]
                                cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 0, 255), 3) # Thicker rectangle for match
                                cv2.putText(processed_image, f"Match: {matched_sample_name} ({match_similarity:.2f})",
                                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                print(f"  MATCH FOUND: Scan face #{j+1} in '{file_name}' matches sample from '{matched_sample_name}' (Similarity: {match_similarity:.4f})")
                                image_had_match = True
                                matches_found_overall = True
                            else:
                                print(f"  No strong match for scan face #{j+1} in '{file_name}' (Highest similarity: {highest_similarity_for_this_face:.4f})")
                    else:
                        print(f"  No faces detected in '{file_name}' to compare.")

                    output_filename = f"{prefix}{'_MATCH' if image_had_match else ''}.jpg"
                    output_path = dataset_output_dir / output_filename
                    print(f"Attempting to save processed dataset image to {output_path}")
                    try:
                        success = cv2.imwrite(str(output_path), processed_image)
                        if success:
                            print(f"Successfully saved processed dataset image to {output_path}")
                        else:
                            print(f"Failed to save processed dataset image to {output_path} (cv2.imwrite returned false)")
                    except Exception as e_write_img:
                        print(f"Error saving processed dataset image {output_filename} to {output_path}: {e_write_img}")

                else:
                    print(f"Failed to process dataset image '{file_name}' (detect_and_encode_faces returned None)")
            except FileNotFoundError:
                print(f"Skipping dataset file '{file_name}' as it was not found or could not be read.")
            except Exception as e:
                print(f"Error processing dataset image '{file_name}' (ID: {file_id}): {str(e)}")
        
        if matches_found_overall:
            print("\nSUCCESS: Matching faces were found and processed images are saved.")
        else:
            print("\nINFO: No matching faces were found in the dataset folder based on the provided samples and threshold.")

def main():
    parser = argparse.ArgumentParser(description='Process folders for face detection and matching.')
    parser.add_argument('--samples', required=True, help='Path or URL/ID of the folder containing sample faces')
    parser.add_argument('--dataset', required=True, help='Path or URL/ID of the folder containing images to search through')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5, help='Minimum detection confidence for face detection (default: 0.5)')
    parser.add_argument('--recognition-threshold', type=float, default=0.5, help='Threshold for face recognition similarity (default: 0.5)') # Added for consistency
    parser.add_argument('--local', action='store_true', help='Use local folders instead of Google Drive')
    args = parser.parse_args()
    
    # In CLI mode, we might want to exit if FaceFinder itself fails to initialize critical components.
    # However, the GUI handles this by logging. For CLI, a more direct exit might be desired.
    # For now, keeping behavior consistent with GUI (log and continue if possible).
    try:
        face_finder = FaceFinder(
            min_detection_confidence=args.min_detection_confidence,
            use_local=args.local
        )
        # The threshold for compare_faces is hardcoded at 0.5 in the class method.
        # If we want to make it configurable via CLI arg like min_detection_confidence,
        # it would need to be passed to compare_faces, likely through process_folders or as a member.
        # For now, the default 0.5 in compare_faces is used. The `recognition-threshold` arg added is for future use.
        face_finder.process_folders(args.samples, args.dataset)
    except RuntimeError as e:
        print(f"A critical error occurred: {e}", file=sys.stderr) # Print critical errors to actual stderr
        # sys.exit(1) # Optionally exit for CLI if critical errors occur
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        # sys.exit(1)


if __name__ == "__main__":
    main()
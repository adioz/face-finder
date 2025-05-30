import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import filedialog
import threading
import sys # For sys.stdout redirection
from face_finder import FaceFinder, TeeLogger # Assuming face_finder.py is in the same directory or accessible
from PIL import Image, ImageTk
import os
from pathlib import Path

# Custom stream object to redirect stdout to the Tkinter text widget
class TextRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, str_val):
        self.widget.insert(tk.END, str_val)
        self.widget.see(tk.END)  # Scroll to the end

    def flush(self):
        pass  # Tkinter text widget is updated in write method

def main():
    # Create the main window
    root = tk.Tk()
    root.title("Face Finder GUI")

    # Create a frame for better organization
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Sample Folder
    sample_folder_label = ttk.Label(main_frame, text="Sample Folder:")
    sample_folder_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    sample_folder_entry = ttk.Entry(main_frame, width=40)
    sample_folder_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
    sample_folder_button = ttk.Button(main_frame, text="Browse...", command=lambda: select_folder(sample_folder_entry))
    sample_folder_button.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)

    # Dataset Folder
    dataset_folder_label = ttk.Label(main_frame, text="Dataset Folder:")
    dataset_folder_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    dataset_folder_entry = ttk.Entry(main_frame, width=40)
    dataset_folder_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
    dataset_folder_button = ttk.Button(main_frame, text="Browse...", command=lambda: select_folder(dataset_folder_entry))
    dataset_folder_button.grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)

    # Min Detection Confidence
    confidence_label = ttk.Label(main_frame, text="Min Detection Confidence:")
    confidence_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    confidence_spinbox = ttk.Spinbox(main_frame, from_=0.1, to=1.0, increment=0.05, width=10)
    confidence_spinbox.set(0.5) # Default value
    confidence_spinbox.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

    # Logs Area (define before start_button so it's available in start_processing_thread)
    logs_label = ttk.Label(main_frame, text="Logs:")
    logs_label.grid(row=4, column=0, sticky=tk.W, padx=5, pady=5) # Adjusted row
    logs_area = scrolledtext.ScrolledText(main_frame, width=60, height=10, wrap=tk.WORD) # Increased width
    logs_area.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5) # Adjusted row

    # Processed Image Display Area
    image_display_frame = ttk.LabelFrame(main_frame, text="Processed Image Viewer", padding="10")
    image_display_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5) # Adjusted row

    image_list_label = ttk.Label(image_display_frame, text="Processed Images:")
    image_list_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

    # Using a Listbox for image selection
    image_listbox = tk.Listbox(image_display_frame, height=5, width=50) # Adjusted height
    image_listbox.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
    image_display_frame.rowconfigure(1, weight=1) # Allow listbox to expand vertically

    image_display_label = ttk.Label(image_display_frame, text="Select an image to display.")
    image_display_label.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

    image_display_frame.columnconfigure(1, weight=1) # Allow image label to expand

    # Start Processing Button - Pass logs_area and other widgets
    start_button = ttk.Button(main_frame, text="Start Processing",
                              command=lambda: threading.Thread(target=start_processing_thread, args=(
                                  sample_folder_entry,
                                  dataset_folder_entry,
                                  confidence_spinbox, # Pass confidence spinbox
                                  logs_area,
                                  start_button,
                                  image_listbox,
                                  image_display_label
                              ), daemon=True).start())
    start_button.grid(row=3, column=0, columnspan=3, pady=10) # Adjusted row

    # Configure column weights for main_frame
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(6, weight=1) # Allow image display frame to expand (adjusted row)

    # Bind selection in Listbox to display function
    image_listbox.bind('<<ListboxSelect>>', lambda event: on_image_select(event, image_listbox, image_display_label))


def select_folder(entry_widget):
    folder_path = filedialog.askdirectory()
    if folder_path:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, folder_path)

def on_image_select(event, listbox_widget, image_label_widget):
    selection = listbox_widget.curselection()
    if not selection:
        return

    selected_index = selection[0]
    # Assuming listbox_widget.image_paths stores the full paths
    if hasattr(listbox_widget, 'image_paths') and len(listbox_widget.image_paths) > selected_index:
        image_path = listbox_widget.image_paths[selected_index]
        display_image(image_path, image_label_widget)
    else: # Fallback if full paths are not stored (e.g. just names)
        filename = listbox_widget.get(selected_index)
        # This case should ideally not be reached if image_paths is always populated correctly.
        # If it is, it means there's a mismatch. For now, display an error on the image label.
        image_label_widget.config(text=f"Error: Full path for {filename} not found.")
        image_label_widget.image = None


def display_image(image_path, image_label_widget):
    try:
        img = Image.open(image_path)

        # Resize image (e.g., max width/height 400px, preserving aspect ratio)
        max_size = (400, 400)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        photo_img = ImageTk.PhotoImage(img)

        image_label_widget.config(image=photo_img, text="") # Clear placeholder text
        image_label_widget.image = photo_img  # Keep a reference!
    except Exception as e:
        error_message = f"Error displaying image {os.path.basename(image_path)}: {e}\n"
        image_label_widget.config(text=error_message) # Show error on label
        image_label_widget.image = None # Clear previous image on error


def populate_image_list(processed_photos_path: Path, listbox_widget, image_label_widget, log_widget):
    listbox_widget.delete(0, tk.END) # Clear existing items
    if hasattr(listbox_widget, 'image_paths'):
        del listbox_widget.image_paths[:] # Clear stored paths
    else:
        listbox_widget.image_paths = []

    image_extensions = {'.jpg', '.jpeg', '.png'}
    found_images = []

    if not processed_photos_path.exists():
        log_widget.insert(tk.END, f"Processed photos directory not found: {processed_photos_path}\n")
        image_label_widget.config(text="Processed photos directory not found.")
        return

    for subdir_name in ['samples', 'dataset']:
        subdir = processed_photos_path / subdir_name
        if subdir.exists():
            for item in sorted(subdir.iterdir()): # Sort for consistent order
                if item.is_file() and item.suffix.lower() in image_extensions:
                    listbox_widget.insert(tk.END, f"{subdir_name} / {item.name}")
                    listbox_widget.image_paths.append(str(item.resolve())) # Store absolute path
                    found_images.append(str(item.resolve()))

    if found_images:
        log_widget.insert(tk.END, f"Populated image list with {len(found_images)} images.\n")
        listbox_widget.selection_set(0) # Select the first item
        display_image(found_images[0], image_label_widget) # Display the first image
    else:
        log_widget.insert(tk.END, "No processed images found to display.\n")
        image_label_widget.config(text="No processed images found.")
        image_label_widget.image = None # Clear any previous image


def start_processing_thread(sample_entry, dataset_entry, confidence_spinbox,
                            log_widget, button_widget, listbox_widget, image_label_widget):
    sample_path = sample_entry.get()
    dataset_path = dataset_entry.get()
    try:
        confidence_value = float(confidence_spinbox.get())
        if not (0.0 <= confidence_value <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
    except ValueError as ve:
        log_widget.insert(tk.END, f"Error: Invalid Min Detection Confidence value: {ve}\n")
        log_widget.see(tk.END)
        return

    if not sample_path or not dataset_path:
        log_widget.insert(tk.END, "Error: Both Sample Folder and Dataset Folder must be selected.\n")
        log_widget.see(tk.END)
        # Clear image list and display if paths are missing
        listbox_widget.delete(0, tk.END)
        if hasattr(listbox_widget, 'image_paths'): listbox_widget.image_paths.clear()
        image_label_widget.config(text="Select an image to display.", image=None)
        image_label_widget.image = None
        return

    # Disable button
    button_widget.config(state=tk.DISABLED)

    # Redirect stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    gui_logger = TextRedirector(log_widget)
    sys.stdout = gui_logger
    sys.stderr = gui_logger

    log_widget.insert(tk.END, "Starting processing...\n")
    log_widget.see(tk.END)

    try:
        log_widget.insert(tk.END, f"Initializing FaceFinder with confidence: {confidence_value}...\n")
        face_finder = FaceFinder(min_detection_confidence=confidence_value, use_local=True)
        log_widget.insert(tk.END, "FaceFinder initialized.\n")

        log_widget.insert(tk.END, f"Processing folders: Sample='{sample_path}', Dataset='{dataset_path}'\n")
        face_finder.process_folders(sample_path, dataset_path)
        log_widget.insert(tk.END, "Processing finished.\n")

        if hasattr(face_finder, 'processed_photos_dir') and face_finder.processed_photos_dir:
            log_widget.insert(tk.END, f"Populating image list from: {face_finder.processed_photos_dir}\n")
            populate_image_list(face_finder.processed_photos_dir, listbox_widget, image_label_widget, log_widget)
        else:
            log_widget.insert(tk.END, "Error: Processed photos directory not available from FaceFinder.\n")
            listbox_widget.delete(0, tk.END)
            if hasattr(listbox_widget, 'image_paths'): listbox_widget.image_paths.clear()
            image_label_widget.config(text="Could not load images: output path missing.", image=None)
            image_label_widget.image = None

    except Exception as e:
        log_widget.insert(tk.END, f"\n--- ERROR DURING PROCESSING ---\n{str(e)}\n-------------------------------\n")
        listbox_widget.delete(0, tk.END) # Clear image list on error
        if hasattr(listbox_widget, 'image_paths'): listbox_widget.image_paths.clear()
        image_label_widget.config(text="Processing failed. See logs for details.", image=None) # Clear image display
        image_label_widget.image = None
    finally:
        # Restore stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        # Re-enable button
        button_widget.config(state=tk.NORMAL)
        log_widget.insert(tk.END, "Processing attempt ended.\n")
        log_widget.see(tk.END)

    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()

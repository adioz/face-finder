import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, PanedWindow
from tkinter.ttk import Style
import threading
import sys
from face_finder import FaceFinder # Assuming face_finder.py is in the same directory or accessible
from PIL import Image, ImageTk
import os
from pathlib import Path

class TextRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, str_val):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, str_val)
        self.widget.see(tk.END)
        self.widget.configure(state='disabled')

    def flush(self):
        pass

def main():
    root = tk.Tk()
    root.title("Face Finder GUI")
    root.geometry("900x700") # Adjusted default size

    style = Style(root)
    try:
        available_themes = style.theme_names()
        if 'clam' in available_themes: style.theme_use('clam')
        elif 'alt' in available_themes: style.theme_use('alt')
        elif 'vista' in available_themes and sys.platform == "win32": style.theme_use('vista')
        elif 'aqua' in available_themes and sys.platform == "darwin": style.theme_use('aqua')
    except Exception:
        print("Error setting theme, using default.")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=0) # Controls area
    root.rowconfigure(1, weight=0) # Start button
    root.rowconfigure(2, weight=1) # Main paned window area

    controls_lf = ttk.LabelFrame(root, text="Controls", padding=(10, 10))
    controls_lf.grid(row=0, column=0, padx=10, pady=(5,5), sticky="ew")
    controls_lf.columnconfigure(1, weight=1)

    path_entries_frame = ttk.Frame(controls_lf, padding=(5,5))
    path_entries_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
    path_entries_frame.columnconfigure(1, weight=1)

    sample_folder_label = ttk.Label(path_entries_frame, text="Sample Folder:")
    sample_folder_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
    sample_folder_entry = ttk.Entry(path_entries_frame, width=60)
    sample_folder_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
    sample_folder_button = ttk.Button(path_entries_frame, text="Browse...", command=lambda: select_folder(sample_folder_entry))
    sample_folder_button.grid(row=0, column=2, sticky=tk.E, padx=5, pady=2)

    dataset_folder_label = ttk.Label(path_entries_frame, text="Dataset Folder:")
    dataset_folder_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
    dataset_folder_entry = ttk.Entry(path_entries_frame, width=60)
    dataset_folder_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
    dataset_folder_button = ttk.Button(path_entries_frame, text="Browse...", command=lambda: select_folder(dataset_folder_entry))
    dataset_folder_button.grid(row=1, column=2, sticky=tk.E, padx=5, pady=2)

    settings_frame = ttk.Frame(controls_lf, padding=(5,5))
    settings_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(2,0))

    confidence_label = ttk.Label(settings_frame, text="Min Detection Confidence:")
    confidence_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
    confidence_spinbox = ttk.Spinbox(settings_frame, from_=0.1, to=1.0, increment=0.05, width=10)
    confidence_spinbox.set(0.5)
    confidence_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

    start_button = ttk.Button(root, text="Start Processing")
    start_button.grid(row=1, column=0, pady=5, padx=10, sticky="ew")

    main_paned_window = PanedWindow(root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=6)
    main_paned_window.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0,10))

    log_lf = ttk.LabelFrame(main_paned_window, text="Logs", padding=(5,5))
    log_lf.columnconfigure(0, weight=1)
    log_lf.rowconfigure(0, weight=1)
    main_paned_window.add(log_lf, weight=1)

    logs_area = scrolledtext.ScrolledText(log_lf, width=45, height=15, wrap=tk.WORD, state='disabled')
    logs_area.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

    right_pane_frame = ttk.Frame(main_paned_window) # No direct padding
    right_pane_frame.columnconfigure(0, weight=1)
    right_pane_frame.rowconfigure(0, weight=1) # Image viewer takes more space initially
    right_pane_frame.rowconfigure(1, weight=1) # Match summary
    main_paned_window.add(right_pane_frame, weight=2)

    image_viewer_lf = ttk.LabelFrame(right_pane_frame, text="Image Viewer", padding=(5,5))
    image_viewer_lf.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
    image_viewer_lf.columnconfigure(0, weight=1); image_viewer_lf.columnconfigure(1, weight=2)
    image_viewer_lf.rowconfigure(1, weight=1)

    image_list_label = ttk.Label(image_viewer_lf, text="Processed Images:")
    image_list_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0,2))

    image_listbox = tk.Listbox(image_viewer_lf, height=6, width=25)
    image_listbox.grid(row=1, column=0, sticky="nsew", padx=(5,2), pady=(0,5))

    image_display_label = ttk.Label(image_viewer_lf, text="Select an image to display.", anchor="center", relief="groove")
    image_display_label.grid(row=1, column=1, sticky="nsew", padx=(2,5), pady=(0,5))

    match_summary_lf = ttk.LabelFrame(right_pane_frame, text="Match Summary", padding=(5,5))
    match_summary_lf.grid(row=1, column=0, sticky="nsew", padx=0, pady=(5,0))
    match_summary_lf.columnconfigure(0, weight=1); match_summary_lf.rowconfigure(0, weight=1)

    summary_cols = { "dataset": ("Dataset Img", 120), "sample": ("Matched Sample", 120),
                     "similarity": ("Similarity", 70), "path": ("Processed Path", 150) }
    summary_treeview = ttk.Treeview(match_summary_lf, columns=list(summary_cols.keys()), show='headings')

    for col_id, (text, width) in summary_cols.items():
        summary_treeview.heading(col_id, text=text, anchor=tk.W)
        summary_treeview.column(col_id, width=width, anchor=tk.W, minwidth=40)

    summary_vsb = ttk.Scrollbar(match_summary_lf, orient="vertical", command=summary_treeview.yview)
    summary_hsb = ttk.Scrollbar(match_summary_lf, orient="horizontal", command=summary_treeview.xview)
    summary_treeview.configure(yscrollcommand=summary_vsb.set, xscrollcommand=summary_hsb.set)
    summary_treeview.grid(row=0, column=0, sticky="nsew")
    summary_vsb.grid(row=0, column=1, sticky="ns")
    summary_hsb.grid(row=1, column=0, sticky="ew")

    start_button.config(command=lambda: threading.Thread(target=start_processing_thread, args=(
                                  root, sample_folder_entry, dataset_folder_entry,
                                  confidence_spinbox, logs_area, start_button,
                                  image_listbox, image_display_label, summary_treeview
                              ), daemon=True).start())

    image_listbox.bind('<<ListboxSelect>>', lambda event: on_image_select(event, image_listbox, image_display_label, logs_area))
    root.mainloop()

def select_folder(entry_widget):
    folder_path = filedialog.askdirectory()
    if folder_path:
        entry_widget.delete(0, tk.END); entry_widget.insert(0, folder_path)

def on_image_select(event, listbox_widget, image_label_widget, log_widget_ref):
    selection = listbox_widget.curselection()
    if not selection: return

    selected_index = selection[0]
    if hasattr(listbox_widget, 'image_paths') and len(listbox_widget.image_paths) > selected_index:
        image_path = listbox_widget.image_paths[selected_index]
        display_image(image_path, image_label_widget, log_widget_ref)
    else:
        filename = listbox_widget.get(selected_index)
        error_msg = f"Error: Full path for {filename} not found."
        image_label_widget.config(text=error_msg, image=None); image_label_widget.image = None
        if log_widget_ref:
            log_widget_ref.configure(state='normal')
            log_widget_ref.insert(tk.END, error_msg + "\n")
            log_widget_ref.configure(state='disabled')

def display_image(image_path, image_label_widget, log_widget_ref):
    try:
        img = Image.open(image_path)
        lbl_width = image_label_widget.winfo_width()
        lbl_height = image_label_widget.winfo_height()
        if lbl_width < 2 or lbl_height < 2: lbl_width, lbl_height = 350, 350 # Fallback size

        max_size = (lbl_width - 10, lbl_height - 10)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        photo_img = ImageTk.PhotoImage(img)
        image_label_widget.config(image=photo_img, text=""); image_label_widget.image = photo_img
    except Exception as e:
        error_message = f"Error displaying image {os.path.basename(image_path)}: {e}"
        image_label_widget.config(text=error_message, image=None); image_label_widget.image = None
        if log_widget_ref:
            log_widget_ref.configure(state='normal')
            log_widget_ref.insert(tk.END, error_message + "\n")
            log_widget_ref.configure(state='disabled')

def populate_image_list(processed_photos_path: Path, listbox_widget, image_label_widget, log_widget):
    listbox_widget.delete(0, tk.END)
    listbox_widget.image_paths = getattr(listbox_widget, 'image_paths', [])
    listbox_widget.image_paths.clear()

    image_extensions = {'.jpg', '.jpeg', '.png'}
    found_images = []
    log_widget.configure(state='normal')
    if not processed_photos_path.exists():
        log_widget.insert(tk.END, f"Processed photos directory not found: {processed_photos_path}\n")
        image_label_widget.config(text="Processed photos directory not found.", image=None); image_label_widget.image = None
    else:
        for subdir_name in ['samples', 'dataset']:
            subdir = processed_photos_path / subdir_name
            if subdir.exists():
                for item in sorted(subdir.iterdir()):
                    if item.is_file() and item.suffix.lower() in image_extensions:
                        listbox_widget.insert(tk.END, f"{subdir_name} / {item.name}")
                        listbox_widget.image_paths.append(str(item.resolve()))
                        found_images.append(str(item.resolve()))
        if found_images:
            log_widget.insert(tk.END, f"Populated image list with {len(found_images)} images.\n")
            listbox_widget.selection_set(0)
            display_image(found_images[0], image_label_widget, log_widget)
        else:
            log_widget.insert(tk.END, "No processed images found to display.\n")
            image_label_widget.config(text="No processed images found.", image=None); image_label_widget.image = None
    log_widget.configure(state='disabled')

def populate_match_summary_treeview(treeview, match_summary_data):
    for i in treeview.get_children(): treeview.delete(i)
    if not match_summary_data: return
    for match in match_summary_data:
        treeview.insert('', 'end', values=(
            match.get('dataset_image_name', 'N/A'), match.get('sample_image_name', 'N/A'),
            f"{match.get('similarity_score', 0.0):.4f}", match.get('processed_dataset_image_path', 'N/A') ))

def start_processing_thread(root_ref, sample_entry, dataset_entry, confidence_spinbox,
                            log_widget, button_widget, listbox_widget, image_label_widget,
                            summary_treeview_ref):
    sample_path = sample_entry.get(); dataset_path = dataset_entry.get()
    log_widget.configure(state='normal')
    try:
        confidence_value = float(confidence_spinbox.get())
        if not (0.0 <= confidence_value <= 1.0): raise ValueError("Confidence must be between 0.0 and 1.0")
    except ValueError as ve:
        log_widget.insert(tk.END, f"Error: Invalid Min Detection Confidence value: {ve}\n"); log_widget.see(tk.END)
        log_widget.configure(state='disabled'); return

    if not sample_path or not dataset_path:
        log_widget.insert(tk.END, "Error: Both Sample Folder and Dataset Folder must be selected.\n"); log_widget.see(tk.END)
        listbox_widget.delete(0, tk.END)
        if hasattr(listbox_widget, 'image_paths'): listbox_widget.image_paths.clear()
        image_label_widget.config(text="Select an image to display.", image=None); image_label_widget.image = None
        populate_match_summary_treeview(summary_treeview_ref, []) # Clear summary
        log_widget.configure(state='disabled'); return

    log_widget.configure(state='disabled'); button_widget.config(state=tk.DISABLED)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    gui_logger = TextRedirector(log_widget)

    def threaded_work():
        sys.stdout = gui_logger; sys.stderr = gui_logger
        log_widget.configure(state='normal'); log_widget.insert(tk.END, "Starting processing...\n"); log_widget.configure(state='disabled')
        match_summary_result = []
        try:
            log_widget.configure(state='normal'); log_widget.insert(tk.END, f"Initializing FaceFinder with confidence: {confidence_value}...\n"); log_widget.configure(state='disabled')
            face_finder = FaceFinder(min_detection_confidence=confidence_value, use_local=True)
            log_widget.configure(state='normal'); log_widget.insert(tk.END, "FaceFinder initialized.\n"); log_widget.insert(tk.END, f"Processing folders: Sample='{sample_path}', Dataset='{dataset_path}'\n"); log_widget.configure(state='disabled')

            match_summary_result = face_finder.process_folders(sample_path, dataset_path)

            log_widget.configure(state='normal'); log_widget.insert(tk.END, "Processing finished.\n")
            if match_summary_result: log_widget.insert(tk.END, f"Found {len(match_summary_result)} potential match(es).\n")
            log_widget.configure(state='disabled')

            root_ref.after(0, lambda: populate_match_summary_treeview(summary_treeview_ref, match_summary_result))

            if hasattr(face_finder, 'processed_photos_dir') and face_finder.processed_photos_dir:
                log_widget.configure(state='normal'); log_widget.insert(tk.END, f"Populating image list from: {face_finder.processed_photos_dir}\n"); log_widget.configure(state='disabled')
                populate_image_list(face_finder.processed_photos_dir, listbox_widget, image_label_widget, log_widget)
            else:
                log_widget.configure(state='normal'); log_widget.insert(tk.END, "Error: Processed photos directory not available from FaceFinder.\n"); log_widget.configure(state='disabled')
                listbox_widget.delete(0, tk.END)
                if hasattr(listbox_widget, 'image_paths'): listbox_widget.image_paths.clear()
                image_label_widget.config(text="Could not load images: output path missing.", image=None); image_label_widget.image = None
                root_ref.after(0, lambda: populate_match_summary_treeview(summary_treeview_ref, []))
        except Exception as e:
            log_widget.configure(state='normal'); log_widget.insert(tk.END, f"\n--- ERROR DURING PROCESSING ---\n{str(e)}\n-------------------------------\n"); log_widget.configure(state='disabled')
            listbox_widget.delete(0, tk.END)
            if hasattr(listbox_widget, 'image_paths'): listbox_widget.image_paths.clear()
            image_label_widget.config(text="Processing failed. See logs for details.", image=None); image_label_widget.image = None
            root_ref.after(0, lambda: populate_match_summary_treeview(summary_treeview_ref, []))
        finally:
            sys.stdout = original_stdout; sys.stderr = original_stderr
            root_ref.after(0, lambda: button_widget.config(state=tk.NORMAL))
            log_widget.configure(state='normal'); log_widget.insert(tk.END, "Processing attempt ended.\n"); log_widget.see(tk.END); log_widget.configure(state='disabled')

    threading.Thread(target=threaded_work, daemon=True).start()

if __name__ == "__main__":
    main()

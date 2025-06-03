import os
import shutil
from blur_filter_script import run_blur_analysis
from color_filter_script import run_color_analysis
from cell_filter_script import run_edge_detection
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def move_images(source_folder, dest_folder):
    ensure_directory_exists(dest_folder)
    for filename in os.listdir(source_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            src_path = os.path.join(source_folder, filename)
            dest_path = os.path.join(dest_folder, filename)
            shutil.move(src_path, dest_path)
    print(f"Moved images from {source_folder} to {dest_folder}")

def run_filtering(folder_start, pink_threshold, white_threshold, pink_white_threshold, cell_threshold, blurr_threshold, purple_threshold):
    folder_1 = os.path.join(folder_start, "good")
    folder_pink = os.path.join(folder_start, "pink")
    folder_pink_white = os.path.join(folder_start, "pink_white")
    folder_pink_good = os.path.join(folder_pink, "good")
    folder_pink_white_good = os.path.join(folder_pink_white, "good")
    folder_2 = os.path.join(folder_1, "good")
    folder_3 = os.path.join(folder_2, "good")
    folder_bad = os.path.join(folder_2, "bad")
    folder_bad_good = os.path.join(folder_bad, "good")

    print("Step 1: Applying color filter (pink, white, pink and white) on", folder_start)
    run_color_analysis(
        type=0,
        input_folder=folder_start,
        pink_option=True,
        pink_white_option=True,
        white_option=True,
        purple_option=False,
        pink_threshold=pink_threshold,
        pink_white_threshold=pink_white_threshold,
        white_threshold=white_threshold,
        purple_threshold=0.0
    )

    print("Step 2: Applying cell count filter on", folder_pink)
    run_edge_detection(
        image_folder=folder_pink,
        type=0,
        threshold=cell_threshold
    )

    print("Step 2: Applying cell count filter on", folder_pink_white)
    run_edge_detection(
        image_folder=folder_pink_white,
        type=0,
        threshold=cell_threshold
    )

    # Move images from folder_pink_good and folder_pink_white_good to folder_1
    if os.path.exists(folder_pink_good):
        move_images(folder_pink_good, folder_1)
    if os.path.exists(folder_pink_white_good):
        move_images(folder_pink_white_good, folder_1)

    print("Step 3: Applying blur filter on", folder_1)
    run_blur_analysis(
        folder_path=folder_1,
        type=0,
        blur_threshold=blurr_threshold
    )

    print("Step 4: Applying cell count filter on", folder_2)
    run_edge_detection(
        image_folder=folder_2,
        type=0,
        threshold=cell_threshold
    )

    print("Step 5: Applying color filter (purple) on", folder_bad)
    run_color_analysis(
        type=0,
        input_folder=folder_bad,
        pink_option=False,
        pink_white_option=False,
        white_option=False,
        purple_option=True,
        pink_threshold=0.0,
        pink_white_threshold=0.0,
        white_threshold=0.0,
        purple_threshold=purple_threshold
    )

    # Move images from folder_bad_good to folder_3
    if os.path.exists(folder_bad_good):
        move_images(folder_bad_good, folder_3)

    final_folder = os.path.join(folder_2, "final")
    if os.path.exists(folder_3):
        os.rename(folder_3, final_folder)
        folder_3 = final_folder

    print("Filtering complete. Final good images are in", folder_3)

# GUI for inputting parameters
class FullFilterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Automated Image Filtering Configuration")
        self.root.geometry("500x500")

        # Default values
        self.defaults = {
            "folder_start": "",
            "pink_threshold": "51",
            "white_threshold": "36",
            "pink_white_threshold": "64",
            "cell_threshold": "25",
            "blurr_threshold": "215",
            "purple_threshold": "62"
        }

        self.entries = {}

        # Create main frame
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        row = 0
        ttk.Label(self.frame, text="Input Folder").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.entries["folder_start"] = ttk.Entry(self.frame, width=30)
        self.entries["folder_start"].insert(0, self.defaults["folder_start"])
        self.entries["folder_start"].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(self.frame, text="Browse", command=self.browse_folder).grid(row=row, column=2, padx=5)
        row += 1

        ttk.Label(self.frame, text="Thresholds", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        threshold_labels = {
            "pink_threshold": "Pink Threshold",
            "white_threshold": "White Threshold",
            "pink_white_threshold": "Pink and White Threshold",
            "cell_threshold": "Cell Count Threshold",
            "blurr_threshold": "Blur Threshold",
            "purple_threshold": "Purple Threshold"
        }

        for param, label in threshold_labels.items():
            ttk.Label(self.frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
            self.entries[param] = ttk.Entry(self.frame)
            self.entries[param].insert(0, self.defaults[param])
            self.entries[param].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            row += 1

        ttk.Button(self.frame, text="Run Filtering", command=self.run_filtering).grid(row=row, column=0, columnspan=3, pady=20)

    def browse_folder(self):
        path = filedialog.askdirectory(title="Select Dataset Folder")
        if path:
            self.entries["folder_start"].delete(0, tk.END)
            self.entries["folder_start"].insert(0, path)

    def run_filtering(self):
        try:
            params = {}
            for param in self.entries:
                value = self.entries[param].get()
                if not value:
                    raise ValueError(f"{param.replace('_', ' ').title()} cannot be empty")
                if param != "folder_start":
                    params[param] = float(value)
                    if params[param] < 0:
                        raise ValueError(f"{param.replace('_', ' ').title()} must be non-negative")
                else:
                    params[param] = value

            if not os.path.exists(params["folder_start"]):
                raise ValueError("Dataset folder does not exist")

            self.root.destroy()
            run_filtering(**params)

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = FullFilterGUI(root)
    root.mainloop()
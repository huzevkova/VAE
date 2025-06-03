import os
import unicodedata
import cv2
import shutil
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from imutils import paths
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

def imread_unicode(path):
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)

# GUI for inputting parameters
class BlurAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Blur Analysis Configuration")
        self.root.geometry("400x200")

        # Default values
        self.defaults = {
            "folder_path": "",
            "type": "0",
            "blur_threshold": "215.0",
            "range_start": "50",
            "range_end": "500",
            "range_step": "5"
        }

        self.entries = {}
        self.labels = {}
        self.widgets = {}

        # Create main frame
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        row = 0
        ttk.Label(self.frame, text="Run Type").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.entries["type"] = ttk.Combobox(self.frame, values=["0 (Filtering)", "1 (Optimizer)"], state="readonly")
        self.entries["type"].set("0 (Filtering)")
        self.entries["type"].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        self.entries["type"].bind("<<ComboboxSelected>>", self.update_inputs)
        row += 1

        ttk.Label(self.frame, text="Input folder").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.entries["folder_path"] = ttk.Entry(self.frame, width=30)
        self.entries["folder_path"].insert(0, self.defaults["folder_path"])
        self.entries["folder_path"].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(self.frame, text="Browse", command=self.browse_folder).grid(row=row, column=2, padx=5)
        row += 1

        self.input_frame = ttk.Frame(self.frame)
        self.input_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        row += 1

        self.update_inputs(None)

        ttk.Button(self.frame, text="Run Analysis", command=self.run_analysis).grid(row=row, column=0, columnspan=3, pady=20)

    def browse_folder(self):
        path = filedialog.askdirectory(title="Select Folder")
        if path:
            self.entries["folder_path"].delete(0, tk.END)
            self.entries["folder_path"].insert(0, path)

    def update_inputs(self, event):
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        self.widgets.clear()

        type_val = self.entries["type"].get()
        row = 0

        if "0" in type_val:
            ttk.Label(self.input_frame, text="Blur Threshold").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.widgets["blur_threshold"] = ttk.Entry(self.input_frame)
            self.widgets["blur_threshold"].insert(0, self.defaults["blur_threshold"])
            self.widgets["blur_threshold"].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        else:
            ttk.Label(self.input_frame, text="Threshold Range Start").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.widgets["range_start"] = ttk.Entry(self.input_frame)
            self.widgets["range_start"].insert(0, self.defaults["range_start"])
            self.widgets["range_start"].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            row += 1

            ttk.Label(self.input_frame, text="Threshold Range End").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.widgets["range_end"] = ttk.Entry(self.input_frame)
            self.widgets["range_end"].insert(0, self.defaults["range_end"])
            self.widgets["range_end"].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            row += 1

            ttk.Label(self.input_frame, text="Threshold Range Step").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.widgets["range_step"] = ttk.Entry(self.input_frame)
            self.widgets["range_step"].insert(0, self.defaults["range_step"])
            self.widgets["range_step"].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)

    def run_analysis(self):
        try:
            params = {}
            params["folder_path"] = self.entries["folder_path"].get()

            type_str = self.entries["type"].get()
            params["type"] = int(type_str[0])

            if params["type"] == 0:
                params["blur_threshold"] = float(self.widgets["blur_threshold"].get())
            else:
                start = int(self.widgets["range_start"].get())
                end = int(self.widgets["range_end"].get())
                step = int(self.widgets["range_step"].get())
                params["threshold_range"] = range(start, end + 1, step)

            if not params["folder_path"]:
                raise ValueError("Folder path cannot be empty")
            if not os.path.exists(params["folder_path"]):
                raise ValueError("Folder path does not exist")
            if params["type"] == 0:
                if params["blur_threshold"] <= 0:
                    raise ValueError("Blur threshold must be positive")
            else:
                if params["threshold_range"].start <= 0:
                    raise ValueError("Range start must be positive")
                if params["threshold_range"].stop <= params["threshold_range"].start:
                    raise ValueError("Range end must be greater than start")
                if params["threshold_range"].step <= 0:
                    raise ValueError("Range step must be positive")

            self.root.destroy()
            run_blur_analysis(**params)

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def optimizer(folder_path, threshold_range):
    sensitivity_list = []
    specificity_list = []
    accuracy_list = []
    thresholds = []

    original_class = {}

    imagePaths = list(paths.list_images(folder_path))

    for imagePath in imagePaths:
        filename = os.path.basename(imagePath).lower()
        if "good" in filename:
            original_class[imagePath] = "not_blurry"
        elif "bad" in filename:
            original_class[imagePath] = "blurry"

    for t in threshold_range:
        TP = FP = TN = FN = 0
        
        for imagePath in imagePaths:
            normalized_path = unicodedata.normalize('NFC', imagePath)
            image = imread_unicode(imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            fm = variance_of_laplacian(gray)
            
            predicted_label = "not_blurry" if fm > t else "blurry"
            
            actual_label = original_class.get(imagePath, None)
            
            if predicted_label == "not_blurry" and actual_label == "not_blurry":
                TN += 1
            elif predicted_label == "not_blurry" and actual_label == "blurry":
                FP += 1
            elif predicted_label == "blurry" and actual_label == "blurry":
                TP += 1 
            elif predicted_label == "blurry" and actual_label == "not_blurry":
                FN += 1
        
        sensitivity = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0 
        specificity = TN / (TN + FP) * 100 if (TN + FP) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) * 100 if (TP + TN + FP + FN) > 0 else 0
        
        thresholds.append(t)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        accuracy_list.append(accuracy)

    df = pd.DataFrame({
        "Threshold": thresholds,
        "Sensitivity": sensitivity_list,
        "Specificity": specificity_list,
        "Accuracy": accuracy_list
    })
    df.to_csv(os.path.join(folder_path, "blur_analysis.csv"), index=False)
    print("Analysis complete. Results saved to 'blur_analysis.csv'.")
    plot_results(sensitivity_list, specificity_list, accuracy_list, threshold_range)

def filter(folder_path, blur_threshold):
    blurry_folder = os.path.join(folder_path, "bad")
    not_blurry_folder = os.path.join(folder_path, "good")
    
    os.makedirs(blurry_folder, exist_ok=True)
    os.makedirs(not_blurry_folder, exist_ok=True)

    image_files = [
        os.path.join(folder_path, f.name)
        for f in os.scandir(folder_path)
        if f.is_file() and f.name.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    for imagePath in image_files:
        image = imread_unicode(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        fm = variance_of_laplacian(gray)
        
        if fm > blur_threshold:
            dest_folder = not_blurry_folder
        else:
            dest_folder = blurry_folder
        
        shutil.move(imagePath, os.path.join(dest_folder, os.path.basename(imagePath)))

    print(f"Filtering complete. Images moved to {blurry_folder} and {not_blurry_folder}.")

def plot_results(sensitivities, specificities, accuracies, r):
    plt.figure(figsize=(10, 5))
    plt.plot(r, sensitivities, label="Sensitivity", marker='o')
    plt.plot(r, specificities, label="Specificity", marker='s')
    plt.plot(r, accuracies, label="Accuracy", marker='^')
    plt.xlabel("Threshold (t)")
    plt.ylabel("Percentage (%)")
    plt.title("Image Classification Metrics")
    plt.legend()
    plt.grid()
    plt.show()


def run_blur_analysis(folder_path, type, blur_threshold=None, threshold_range=None):
    if type == 1:
        optimizer(folder_path, threshold_range)
    else:
        filter(folder_path, blur_threshold)

if __name__ == "__main__":
    root = tk.Tk()
    app = BlurAnalysisGUI(root)
    root.mainloop()
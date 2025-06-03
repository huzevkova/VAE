import os
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
import csv
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

def imread_unicode(path):
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def sobel_edge_detection(image, verbose=False):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) 
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3) 
    
    gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    
    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()
    
    return gradient_magnitude.astype(np.uint8)

def sobel_red_count(image_file):
    image = imread_unicode(image_file)

    if image is None:
        raise FileNotFoundError("The image was not found. Please check the file path.")

    edges = sobel_edge_detection(image)

    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(edges, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 11)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    spot_count = 0
    dark_threshold = 85  # Max intensity for dark spots
    max_aspect_ratio = 3  # Ignore ellipses that are too stretched

    for contour in contours:
        area = cv2.contourArea(contour)
        
        if 5 < area < 200:
            mask = np.zeros_like(image, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
          
            mean_intensity = cv2.mean(image, mask=mask)[0]
            
            if mean_intensity <= dark_threshold:
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    (x, y), (major_axis, minor_axis), angle = ellipse

                    aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)

                    if aspect_ratio <= max_aspect_ratio:
                        spot_count += 1
                        cv2.ellipse(image_color, ellipse, (0, 0, 255), 1)
                
                else: 
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    cv2.circle(image_color, (int(x), int(y)), int(radius), (0, 0, 255), 1)
                    spot_count += 1

    return spot_count

def filter_images(image_folder, good_folder, bad_folder, threshold):
    ensure_directory_exists(good_folder)
    ensure_directory_exists(bad_folder)

    image_files = [
        os.path.join(image_folder, f.name)
        for f in os.scandir(image_folder)
        if f.is_file() and f.name.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    for image_path in image_files:
        try:
            cell_count = sobel_red_count(image_path)
            filename = os.path.basename(image_path)
            if cell_count >= threshold:
                shutil.move(image_path, os.path.join(good_folder, filename))
            else:
                shutil.move(image_path, os.path.join(bad_folder, filename))
        except FileNotFoundError as e:
            print(f"Error processing {image_path}: {e}")
    
    print(f"Filtering complete. Images moved to {good_folder} and {bad_folder}.")

def optimizer(image_folder, threshold_range, sensitivities, specificities, accuracies):
    for t in threshold_range:
        tp = tn = fp = fn = 0

        for filename in os.listdir(image_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)
                try:
                    cell_count = sobel_red_count(image_path)
                    
                    if cell_count >= t:
                        if "good" in filename.lower():
                            tn += 1
                        else:
                            fn += 1
                    else:
                        if "good" in filename.lower():
                            fp += 1
                        else:
                            tp += 1
                except FileNotFoundError as e:
                    print(f"Error processing {image_path}: {e}")

        specificity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        sensitivity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        accuracy = (tn + tp) / (tn + tp + fp + fn) * 100 if (tn + tp + fp + fn) > 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)
        accuracies.append(accuracy)

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


def export_to_csv(sensitivities, specificities, accuracies, threshold_range):
    name = "cell_count_analysis.csv"
    with open(name, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Threshold", "Sensitivity", "Specificity", "Accuracy"])
        for i, t in enumerate(threshold_range):
            writer.writerow([t, sensitivities[i], specificities[i], accuracies[i]])

# GUI for inputting parameters
class EdgeDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Edge Detection Configuration")
        self.root.geometry("400x300")

        # Default values
        self.defaults = {
            "image_folder": "",
            "type": "0", 
            "threshold": "25",
            "range_start": "0",
            "range_end": "50",
            "range_step": "1"
        }

        self.entries = {}
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

        ttk.Label(self.frame, text="Image Folder").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.entries["image_folder"] = ttk.Entry(self.frame, width=30)
        self.entries["image_folder"].insert(0, self.defaults["image_folder"])
        self.entries["image_folder"].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(self.frame, text="Browse", command=self.browse_folder).grid(row=row, column=2, padx=5)
        row += 1

        self.input_frame = ttk.Frame(self.frame)
        self.input_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        row += 1

        self.update_inputs(None)

        ttk.Button(self.frame, text="Run Analysis", command=self.run_analysis).grid(row=row, column=0, columnspan=3, pady=20)

    def browse_folder(self):
        path = filedialog.askdirectory(title="Select Image Folder")
        if path:
            self.entries["image_folder"].delete(0, tk.END)
            self.entries["image_folder"].insert(0, path)

    def update_inputs(self, event):
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        self.widgets.clear()

        type_val = self.entries["type"].get()
        row = 0

        if "0" in type_val:
            ttk.Label(self.input_frame, text="Threshold").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.widgets["threshold"] = ttk.Entry(self.input_frame)
            self.widgets["threshold"].insert(0, self.defaults["threshold"])
            self.widgets["threshold"].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
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
            params["image_folder"] = self.entries["image_folder"].get()
            type_str = self.entries["type"].get()
            params["type"] = int(type_str[0])

            if not params["image_folder"]:
                raise ValueError("Image folder cannot be empty")
            if not os.path.exists(params["image_folder"]):
                raise ValueError("Image folder does not exist")

            if params["type"] == 0:
                params["threshold"] = float(self.widgets["threshold"].get())
                if params["threshold"] < 0:
                    raise ValueError("Threshold must be non-negative")
            else:
                start = int(self.widgets["range_start"].get())
                end = int(self.widgets["range_end"].get())
                step = int(self.widgets["range_step"].get())
                if start < 0:
                    raise ValueError("Range start must be non-negative")
                if end <= start:
                    raise ValueError("Range end must be greater than start")
                if step <= 0:
                    raise ValueError("Range step must be positive")
                params["threshold_range"] = range(start, end + 1, step)

            self.root.destroy()
            run_edge_detection(**params)

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

def run_edge_detection(image_folder, type, threshold=None, threshold_range=None):
    print(image_folder)
    if type == 0:
        good_folder = os.path.join(image_folder, "good")
        bad_folder = os.path.join(image_folder, "bad")
        filter_images(image_folder, good_folder, bad_folder, threshold)
    else:
        sensitivities = []
        specificities = []
        accuracies = []
        optimizer(image_folder, threshold_range, sensitivities, specificities, accuracies)
        export_to_csv(sensitivities, specificities, accuracies, threshold_range)
        plot_results(sensitivities, specificities, accuracies, threshold_range)

if __name__ == "__main__":
    root = tk.Tk()
    app = EdgeDetectionGUI(root)
    root.mainloop()
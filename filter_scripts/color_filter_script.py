import os
import cv2
import shutil
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import webcolors
import csv
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

NUMBER_OF_GROUPS = 7

# GUI for inputting parameters
class ColorAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Analysis Configuration")
        self.root.geometry("450x500")

        # Default values
        self.defaults = {
            "input_folder": "",
            "image_path": "",
            "type": "0",
            "pink_threshold": "51",
            "pink_white_threshold": "36",
            "white_threshold": "64",
            "purple_threshold": "0"
        }

        self.entries = {}
        self.widgets = {}
        self.color_vars = {}

        # Create main frame
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        row = 0
        ttk.Label(self.frame, text="Run Type").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.entries["type"] = ttk.Combobox(self.frame, values=["0 (Filtering)", "1 (Optimizer)", "2 (One Image Analysis)"], state="readonly")
        self.entries["type"].set("0 (Filtering)")
        self.entries["type"].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        self.entries["type"].bind("<<ComboboxSelected>>", self.update_inputs)
        row += 1

        self.input_frame = ttk.Frame(self.frame)
        self.input_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        row += 1

        self.update_inputs(None)

        ttk.Button(self.frame, text="Run Analysis", command=self.run_analysis).grid(row=row, column=0, columnspan=3, pady=20)

    def update_inputs(self, event):
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        self.widgets.clear()
        self.color_vars.clear()

        type_val = self.entries["type"].get()
        row = 0

        if "0" in type_val or "1" in type_val:
            ttk.Label(self.input_frame, text="Input Folder").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.widgets["input_folder"] = ttk.Entry(self.input_frame, width=30)
            self.widgets["input_folder"].insert(0, self.defaults["input_folder"])
            self.widgets["input_folder"].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            ttk.Button(self.input_frame, text="Browse", command=self.browse_folder).grid(row=row, column=2, pady = 5)
            row += 1

            row += 1
            ttk.Label(self.input_frame, text="Select Colors").grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
            row += 1

            colors = ["pink", "pink_white", "white", "purple"]
            if "0" in type_val:
                for color in colors:
                    self.color_vars[color] = tk.BooleanVar(value=False)
                    ttk.Checkbutton(self.input_frame, text=color.capitalize(), variable=self.color_vars[color]).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
                    if (color != "purple"):
                        self.color_vars[color].set(True)
                    else:
                        self.color_vars[color].set(False)
                    row += 1
                    ttk.Label(self.input_frame, text=f"{color.capitalize()} Threshold").grid(row=row, column=0, sticky=tk.W, pady=2)
                    self.widgets[f"{color}_threshold"] = ttk.Entry(self.input_frame)
                    self.widgets[f"{color}_threshold"].insert(0, self.defaults[f"{color}_threshold"])
                    self.widgets[f"{color}_threshold"].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
                    row += 1
            else:
                self.color_vars["selected_color"] = tk.StringVar(value=colors[0])
                for color in colors:
                    ttk.Radiobutton(self.input_frame, text=color.capitalize(), value=color, variable=self.color_vars["selected_color"]).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
                    row += 1

        else:
            ttk.Label(self.input_frame, text="Image Path").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.widgets["image_path"] = ttk.Entry(self.input_frame, width=30)
            self.widgets["image_path"].insert(0, self.defaults["image_path"])
            self.widgets["image_path"].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            ttk.Button(self.input_frame, text="Browse", command=self.browse_image).grid(row=row, column=2, padx=5)
            row += 1

    def browse_folder(self):
        path = filedialog.askdirectory(title="Select Input Folder")
        if path:
            self.widgets["input_folder"].delete(0, tk.END)
            self.widgets["input_folder"].insert(0, path)

    def browse_image(self):
        path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if path:
            self.widgets["image_path"].delete(0, tk.END)
            self.widgets["image_path"].insert(0, path)

    def run_analysis(self):
        try:
            params = {}
            type_str = self.entries["type"].get()
            params["type"] = int(type_str[0])

            if params["type"] in [0, 1]:
                params["input_folder"] = self.widgets["input_folder"].get()
                if not params["input_folder"]:
                    raise ValueError("Input folder cannot be empty")
                if not os.path.exists(params["input_folder"]):
                    raise ValueError("Input folder does not exist")

                if params["type"] == 0:
                    params["pink_option"] = self.color_vars["pink"].get()
                    params["pink_white_option"] = self.color_vars["pink_white"].get()
                    params["white_option"] = self.color_vars["white"].get()
                    params["purple_option"] = self.color_vars["purple"].get()

                    if not any([params["pink_option"], params["pink_white_option"], params["white_option"], params["purple_option"]]):
                        raise ValueError("At least one color must be selected")

                    for color in ["pink", "pink_white", "white"]:
                        if self.color_vars[color].get():
                            params[f"{color}_threshold"] = float(self.widgets[f"{color}_threshold"].get())
                            if params[f"{color}_threshold"] < 0:
                                raise ValueError(f"{color.capitalize()} threshold must be non-negative")
                    params["purple_threshold"] = 0.0

                else:
                    params["selected_color"] = self.color_vars["selected_color"].get()
                    for color in ["pink", "pink_white", "white", "purple"]:
                        params[f"{color}_option"] = (color == params["selected_color"])

            else:
                params["image_path"] = self.widgets["image_path"].get()
                if not params["image_path"]:
                    raise ValueError("Image path cannot be empty")
                if not os.path.exists(params["image_path"]):
                    raise ValueError("Image path does not exist")

            self.root.destroy()
            run_color_analysis(**params)

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

def closest_color(rgb):
    """Finds the closest named color for an RGB value."""
    min_colors = {}
    for name in webcolors.names("css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - rgb[0]) ** 2
        gd = (g_c - rgb[1]) ** 2
        bd = (b_c - rgb[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def classify_named_color(color_name):
    #Maps known color names to pink, purple, white, or other
    pink_shades = {"lightpink", "hotpink", "deeppink", "palevioletred", "mediumvioletred", "orchid", "thistle", "plum"}
    purple_shades = {"grey", "dimgrey", "rosybrown", "mediumorchid", "purple", "indigo", "slategrey", "darkmagenta", "darkviolet", "darkslateblue", "blueviolet", "darkorchid", "slateblue", "mediumslateblue", "mediumorchid", "mediumpurple", "violet", "lavender", "midnightblue", "navy", "darkblue", "mediumblue", "blue", "royalblue", "steelblue", "dodgerblue", "deepskyblue", "cornflowerblue", "skyblue", "lightskyblue"}
    white_shades = {"pink", "white", "darkgrey", "lightslategrey", "silver", "gainsboro", "linen", "whitesmoke", "ghostwhite", "snow", "lavenderblush", "peachpuff", "mistyrose"}

    if color_name in pink_shades:
        return "pink"
    elif color_name in purple_shades:
        return "purple"
    elif color_name in white_shades:
        return "white"
    else:
        return "other"

def analyze_image_colors(image_path, num_groups, tp=0, tpw=0, tw=0, tpp=0, pink=False, pinkWhite=False, white=False, purple=False, single=False):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image_data = np.array(image)

    pixels = image_data.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=num_groups, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    cluster_centers = kmeans.cluster_centers_.astype(int)
    
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    
    total_pixels = pixels.shape[0]
    percentages = (counts / total_pixels) * 100

    pink_percentage = 0
    purple_percentage = 0
    white_percentage = 0

    if single:
        print("Color Group Analysis:")
        for i, (color, percentage) in enumerate(zip(cluster_centers, percentages)):
            color_name = closest_color(color)
            classified_name = classify_named_color(color_name)
            print(f"Group {i+1}: Color RGB {tuple(color)} - {percentage:.2f}% - Closest Color: {color_name} - Category: {classified_name}")
            if classified_name in ["pink", "other"]:
                pink_percentage += percentage
            if classified_name == "purple":
                purple_percentage += percentage
            if classified_name == "white":
                white_percentage += percentage

        transformed_pixels = cluster_centers[kmeans.labels_]
        transformed_image_data = transformed_pixels.reshape(image_data.shape)
        transformed_image = Image.fromarray(transformed_image_data.astype('uint8'))
        transformed_image.show()

        return cluster_centers, percentages
    else:
        for i, (color, percentage) in enumerate(zip(cluster_centers, percentages)):
            color_name = closest_color(color)
            classified_name = classify_named_color(color_name)
            if classified_name == "pink":
                pink_percentage += percentage
            if classified_name == "purple":
                purple_percentage += percentage
            if classified_name in ["white", "other"]:
                white_percentage += percentage

        if white_percentage >= tw and white:
            return "too_white"
        if pink_percentage >= tp and pink:
            return "too_pink"
        if pink_percentage + white_percentage >= tpw and pinkWhite:
            return "too_pink_white"
        if purple:
            if purple_percentage >= tpp:
                return "filtered_images"
            else:
                return "not_enough_purple"
            
        return "filtered_images"

def filter_images_stat(input_folder, t, sensitivities, specificities, accuracies, pink_option, pink_white_option, white_option, purple_option):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            classification = analyze_image_colors(image_path, NUMBER_OF_GROUPS, 
                                               tp=(t if pink_option else 0), 
                                               tpw=(t if pink_white_option else 0), 
                                               tw=(t if white_option else 0), 
                                               tpp=(t if purple_option else 0), 
                                               pink=pink_option, 
                                               pinkWhite=pink_white_option, 
                                               white=white_option, 
                                               purple=purple_option)
            
            if classification == "filtered_images":
                if "good" in filename.lower():
                    tn += 1
                else:
                    fn += 1
            else:
                if "good" in filename.lower():
                    fp += 1
                else:
                    tp += 1
    
    sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    accuracy = (tn + tp) / (tn + tp + fp + fn) * 100 if (tn + tp + fp + fn) > 0 else 0
    
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    accuracies.append(accuracy)

def plot_results(sensitivities, specificities, accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(range(100), sensitivities, label="Sensitivity", marker='o')
    plt.plot(range(100), specificities, label="Specificity", marker='s')
    plt.plot(range(100), accuracies, label="Accuracy", marker='^')
    plt.xlabel("Threshold (t)")
    plt.ylabel("Percentage (%)")
    plt.title("Image Classification Metrics")
    plt.legend()
    plt.grid()
    plt.show()

def export_to_csv(sensitivities, specificities, accuracies, input_folder):
    with open(os.path.join(input_folder, "color_analysis.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Threshold", "Sensitivity", "Specificity", "Accuracy"])
        for i in range(100):
            writer.writerow([i, sensitivities[i], specificities[i], accuracies[i]])

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def filter_images(input_folder, pink_threshold, pink_white_threshold, white_threshold, purple_threshold, pink_option, pink_white_option, white_option, purple_option):
    output_folder = os.path.join(input_folder, "good")
    purple_folder = os.path.join(input_folder, "not_enough_purple")
    pink_folder = os.path.join(input_folder, "pink")
    white_folder = os.path.join(input_folder, "white")
    pink_white_folder = os.path.join(input_folder, "pink_white")

    create_directory(output_folder)

    if (purple_option):
        create_directory(purple_folder)

    if (pink_option):
        create_directory(pink_folder)

    if (white_option):
        create_directory(white_folder)

    if (pink_white_option):
        create_directory(pink_white_folder)    

    for filename in os.scandir(input_folder):
        if filename.is_file() and filename.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = filename.path
            
            classification = analyze_image_colors(image_path, NUMBER_OF_GROUPS, 
                                               tp=pink_threshold, 
                                               tpw=pink_white_threshold, 
                                               tw=white_threshold, 
                                               tpp=purple_threshold, 
                                               pink=pink_option, 
                                               pinkWhite=pink_white_option, 
                                               white=white_option, 
                                               purple=purple_option)
                        
            filename = os.path.basename(image_path)
            if classification == "filtered_images":
                shutil.move(image_path, os.path.join(output_folder, filename))
            else:
                if classification == "not_enough_purple":
                    shutil.move(image_path, os.path.join(purple_folder, filename))
                elif classification == "too_pink":
                    shutil.move(image_path, os.path.join(pink_folder, filename))
                elif classification == "too_white":
                    shutil.move(image_path, os.path.join(white_folder, filename))
                elif classification == "too_pink_white":
                    shutil.move(image_path, os.path.join(pink_white_folder, filename))
    print(f"Filtering complete. Images moved to {output_folder}, {purple_folder}, {pink_folder}, {white_folder}, and {pink_white_folder}.")

def run_color_analysis(type, input_folder=None, image_path=None, pink_option=False, pink_white_option=False, white_option=False, purple_option=False, pink_threshold=0.0, pink_white_threshold=0.0, white_threshold=0.0, purple_threshold=0.0, selected_color=None):
    if type == 1:
        sensitivities = []
        specificities = []
        accuracies = []
        
        for i in range(100):
            filter_images_stat(input_folder, i, sensitivities, specificities, accuracies, 
                             pink_option, pink_white_option, white_option, purple_option)
            print(f"Threshold {i} done")
        
        plot_results(sensitivities, specificities, accuracies)
        export_to_csv(sensitivities, specificities, accuracies, input_folder)
    elif type == 0:
        filter_images(input_folder, pink_threshold, pink_white_threshold, white_threshold, purple_threshold, 
                     pink_option, pink_white_option, white_option, purple_option)
    elif type == 2:
        clusters, percentages = analyze_image_colors(image_path, NUMBER_OF_GROUPS, single=True)

        def plot_color_distribution(cluster_centers, percentages):
            colors = [tuple(color / 255) for color in cluster_centers]
            plt.figure(figsize=(8, 8))
            plt.pie(percentages, labels=[f"Color: {closest_color(color)} - Category: {classify_named_color(closest_color(color))}" for color in cluster_centers],
                    colors=colors, autopct='%1.1f%%', textprops={'fontsize': 14})
            plt.title("Color Distribution")
            plt.show()

        plot_color_distribution(clusters, percentages)

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorAnalysisGUI(root)
    root.mainloop()
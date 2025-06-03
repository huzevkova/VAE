import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import csv
import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# GUI
class VAETrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VAE Training Configuration")
        self.root.geometry("400x600")

        # Default hyperparameter values
        self.defaults = {
            "img_size": "128",
            "latent_dim": "750",
            "batch_size": "10",
            "num_epochs": "25",
            "alpha": "5",
            "beta": "1",
            "gamma": "10",
            "lr": "0.0001",
            "in_channels": "3",
            "depth": "4",
            "file_path": "",
            "log_file": "./vae_training_log.csv",
            "model_save_dir": "./vae_checkpoints",
            "output_dir": "./output_images"
        }

        self.entries = {}

        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        row = 0
        ttk.Label(self.frame, text="Hyperparameters", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        for param, default in list(self.defaults.items())[:10]:  # Hyperparameters only
            ttk.Label(self.frame, text=param).grid(row=row, column=0, sticky=tk.W, pady=2)
            self.entries[param] = ttk.Entry(self.frame)
            self.entries[param].insert(0, default)
            self.entries[param].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            row += 1

        ttk.Label(self.frame, text="Paths", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        for param in ["file_path", "log_file", "model_save_dir", "output_dir"]:
            ttk.Label(self.frame, text=param).grid(row=row, column=0, sticky=tk.W, pady=2)
            self.entries[param] = ttk.Entry(self.frame)
            self.entries[param].insert(0, self.defaults[param])
            self.entries[param].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            ttk.Button(self.frame, text="Browse", command=lambda p=param: self.browse_path(p)).grid(row=row, column=2, padx=5)
            row += 1

        ttk.Button(self.frame, text="Run Training", command=self.run_training).grid(row=row, column=0, columnspan=3, pady=20)

    def browse_path(self, param):
        if param in ["file_path", "model_save_dir", "output_dir"]:
            path = filedialog.askdirectory(title=f"Select {param}")
        else: 
            path = filedialog.asksaveasfilename(title="Select log file", defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            self.entries[param].delete(0, tk.END)
            self.entries[param].insert(0, path)

    def run_training(self):
        try:
            params = {}
            for param, entry in self.entries.items():
                value = entry.get()
                if param in ["img_size", "latent_dim", "batch_size", "num_epochs", "in_channels", "depth"]:
                    params[param] = int(value)
                elif param in ["alpha", "beta", "gamma", "lr"]:
                    params[param] = float(value)
                else:  
                    params[param] = value

            for param, value in params.items():
                if not value:
                    raise ValueError(f"{param} cannot be empty")
                if isinstance(value, (int, float)) and value <= 0:
                    raise ValueError(f"{param} must be positive")

            self.root.destroy()
            run_vae_training(**params)

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

# VAE
def run_vae_training(img_size, latent_dim, batch_size, num_epochs, alpha, beta, gamma, lr, in_channels, depth, file_path, log_file, model_save_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset loading
    class MedicalImageDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith('.jpg')]

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image

    # Data transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load dataset
    dataset = MedicalImageDataset(file_path, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)

    class VAE(nn.Module):
        def __init__(self, in_channels=3, latent_dim=128, depth=4):
            super(VAE, self).__init__()
            self.depth = depth
            self.latent_dim = latent_dim

            self.encoder_channels = [in_channels] + [32 * (2**i) for i in range(depth)]
            self.decoder_channels = self.encoder_channels[::-1] 
            self.feature_size = img_size // (2**depth) 

            # Encoder
            encoder_layers = []
            for i in range(depth):
                encoder_layers.append(
                    nn.Conv2d(
                        self.encoder_channels[i],
                        self.encoder_channels[i+1],
                        kernel_size=4,
                        stride=2,
                        padding=1
                    )
                )
                encoder_layers.append(nn.ReLU())
            self.encoder = nn.Sequential(*encoder_layers)

            self.fc_mu = nn.Linear(self.encoder_channels[-1] * self.feature_size * self.feature_size, latent_dim)
            self.fc_logvar = nn.Linear(self.encoder_channels[-1] * self.feature_size * self.feature_size, latent_dim)
            self.fc_decode = nn.Linear(latent_dim, self.decoder_channels[0] * self.feature_size * self.feature_size)

            # Decoder
            decoder_layers = []
            for i in range(depth):
                decoder_layers.append(
                    nn.ConvTranspose2d(
                        self.decoder_channels[i],
                        self.decoder_channels[i+1],
                        kernel_size=4,
                        stride=2,
                        padding=1
                    )
                )
                decoder_layers.append(nn.ReLU() if i < depth-1 else nn.Tanh())
            self.decoder = nn.Sequential(*decoder_layers)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            x = self.encoder(x)
            x = x.view(x.size(0), -1)
            mu, logvar = self.fc_mu(x), self.fc_logvar(x)
            z = self.reparameterize(mu, logvar)
            x = self.fc_decode(z).view(x.size(0), self.decoder_channels[0], self.feature_size, self.feature_size)
            x = self.decoder(x)
            return x, mu, logvar

    # Transform image and calculate laplacian
    def laplacian_variance(image):
        if image.dim() == 4:
            image = image.mean(dim=1)

        image_np = image.cpu().detach().numpy()

        laplacian_var_list = []
        for img in image_np:
            img = np.uint8(np.clip(img * 255, 0, 255)) 

            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            variance = np.var(laplacian)
            laplacian_var_list.append(variance)

        return torch.tensor(laplacian_var_list, device=image.device, dtype=torch.float32)

    # Calculate loss
    def loss_function(recon_x, x, mu, logvar, blur_threshold=215.0):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        recon_blur_var = laplacian_variance(recon_x)
        blur_loss = F.relu(blur_threshold - recon_blur_var).sum()

        total_loss = alpha * recon_loss + beta * kl_div + gamma * blur_loss
        return total_loss, recon_loss.item(), kl_div.item(), blur_loss.item()

    # Load the model from the last checkpoint
    def load_checkpoint(model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, start_epoch, loss

    # Training
    vae = VAE(in_channels=in_channels, latent_dim=latent_dim, depth=depth).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device='cuda')

    # Check if checkpoint exists
    latest_checkpoint = None
    if os.path.exists(model_save_dir):
        for filename in os.listdir(model_save_dir):
            if filename.endswith(".pth"):
                latest_checkpoint = os.path.join(model_save_dir, filename)

    start_epoch = 0
    if latest_checkpoint:
        print(f"Loading checkpoint from {latest_checkpoint}")
        vae, optimizer, start_epoch, _ = load_checkpoint(vae, optimizer, latest_checkpoint)
    else:
        print("No checkpoint found, starting from scratch")

    # Initialize log data for epochs and loss
    log_data = {
        "Epoch": [],
        "Total Loss": [],
        "Reconstruction Loss": [],
        "KL Divergence": [],
        "Blur Loss": []
    }

    # Define hyperparameters to log
    hyperparams = {
        "img_size": img_size,
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "lr": lr,
        "in_channels": in_channels,
        "depth": depth
    }

    # Write hyperparameters to CSV
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(hyperparams.keys()) 
        writer.writerow(hyperparams.values())
        writer.writerow([]) 
        writer.writerow(["Epoch", "Loss"])

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        vae.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_blur = 0

        print("Epoch start")
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "hip" if torch.version.hip else "cpu"):
                recon_batch, mu, logvar = vae(batch)
                loss, recon_l, kl_l, blur_l = loss_function(recon_batch, batch, mu, logvar)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track per-batch losses (optional: for batch-level logging)
            total_loss += loss.item()
            total_recon += recon_l
            total_kl += kl_l
            total_blur += blur_l

        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon = total_recon / len(train_loader.dataset)
        avg_kl = total_kl / len(train_loader.dataset)
        avg_blur = total_blur / len(train_loader.dataset)

        log_data["Epoch"].append(epoch + 1)
        log_data["Total Loss"].append(avg_loss)
        log_data["Reconstruction Loss"].append(avg_recon)
        log_data["KL Divergence"].append(avg_kl)
        log_data["Blur Loss"].append(avg_blur)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        # Save model
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        model_filename = os.path.join(model_save_dir, f"vae_epoch_{epoch+1}.pth")
        torch.save(checkpoint, model_filename)
        print(f"Model checkpoint saved as '{model_filename}'")

        # Append training log to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([epoch + 1, avg_loss])
        print(f"Training log updated: {log_file}")

    # Plot training loss
    plt.plot(log_data["Epoch"], log_data["Total Loss"], label="Total Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("VAE Training Loss")
    plt.show()

    # Generate new images
    vae.eval()
    with torch.no_grad():
        z = torch.randn(64, latent_dim).to(device)
        z = vae.fc_decode(z).view(-1, vae.decoder_channels[0], vae.feature_size, vae.feature_size)
        generated_images = vae.decoder(z).cpu()

    generated_images = (generated_images + 1) / 2
    grid = vutils.make_grid(generated_images, nrow=8)

    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(generated_images):
        img = np.transpose(img.numpy(), (1, 2, 0))
        img = (img * 255).astype(np.uint8)

        pil_img = Image.fromarray(img)
        img_path = os.path.join(output_dir, f'generated_image_{i+1}.png')
        pil_img.save(img_path)

    plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(grid.numpy(), (1,2,0)))
    plt.axis("off")
    plt.title("Generated Images")
    plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = VAETrainingGUI(root)
    root.mainloop()
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyvista as pv
import warnings
import gc

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

GRID_STEPS = n  # Set resolution (10, 50, 100, 200)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
CHANNELS = 1
LATENT_DIM = 100
BATCH_SIZE = 4
SUBSET_SIZE = 8
PERTURB_SCALE = 0.5

GEN_CHECKPOINT = "generator.pth"
DISC_CHECKPOINT = "discriminatorpth"
IMG_FOLDER = "image_folder"

if not os.path.exists(GEN_CHECKPOINT):
    print(f"Generator Checkpoint NOT FOUND: {GEN_CHECKPOINT}")
if not os.path.exists(DISC_CHECKPOINT):
    print(f"Discriminator Checkpoint NOT FOUND: {DISC_CHECKPOINT}")
if not os.path.isdir(IMG_FOLDER):
    print(f"Image Folder NOT FOUND: {IMG_FOLDER}")

class FloorPlanDataset(Dataset):
    def __init__(self, img_folder, transform=None):
        try:
            self.img_files = os.listdir(img_folder)[:SUBSET_SIZE]
            self.img_folder_path = img_folder
            self.transform = transform
        except FileNotFoundError:
            self.img_files, self.img_folder_path = [], ""
            print(f"Warning: Folder '{img_folder}' not found.")

    def __len__(self): return len(self.img_files)
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_folder_path, self.img_files[idx])).convert("L")
        return self.transform(img) if self.transform else img

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = FloorPlanDataset(IMG_FOLDER, transform)
subset_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512*16*16)
        def block(in_f, out_f): 
            return nn.Sequential(nn.BatchNorm2d(in_f),
                                 nn.ConvTranspose2d(in_f, out_f, 4, 2, 1),
                                 nn.ReLU(True))
        self.gen = nn.Sequential(block(512, 256),
                                 block(256, 128),
                                 block(128, 64),
                                 nn.ConvTranspose2d(64, img_channels, 4, 2, 1),
                                 nn.Tanh())
    def forward(self, z): 
        return self.gen(self.fc(z).view(z.size(0), 512, 16, 16))

class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super().__init__()
        def block(in_f, out_f, bn=True):
            layers = [nn.Conv2d(in_f, out_f, 4, 2, 1)]
            if bn: layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(*block(img_channels, 64, bn=False),
                                   *block(64, 128),
                                   *block(128, 256),
                                   *block(256, 512),
                                   nn.Conv2d(512, 1, 4, 1, 0))
    def forward(self, img): return self.model(img)

generator = Generator(LATENT_DIM, CHANNELS).to(DEVICE)
discriminator = Discriminator(CHANNELS).to(DEVICE)

generator.load_state_dict(torch.load(GEN_CHECKPOINT, map_location=DEVICE))
discriminator.load_state_dict(torch.load(DISC_CHECKPOINT, map_location=DEVICE))
generator.eval(); discriminator.eval()

adversarial_loss = nn.BCEWithLogitsLoss()

def get_gan_loss(gen, disc, dataloader, latent_dim):
    total_loss = 0
    with torch.no_grad():
        for imgs in dataloader:
            imgs = imgs.to(DEVICE)
            z = torch.randn(imgs.size(0), latent_dim, device=DEVICE)
            loss = adversarial_loss(disc(gen(z)), torch.ones_like(disc(gen(z))))
            total_loss += loss.item() * imgs.size(0)
    return total_loss / len(dataloader.dataset)

def get_random_directions(model):
    params = [model.gen[3].weight, model.gen[3].bias]
    d1 = [torch.randn_like(p) for p in params]
    d2 = [torch.randn_like(p) for p in params]
    d1 = [d / (d.norm() + 1e-8) for d in d1]
    d2 = [d / (d.norm() + 1e-8) for d in d2]
    return d1, d2

def compute_loss_surface(model, disc, dataloader, d1, d2, steps, scale, latent_dim):
    alphas = np.linspace(-scale, scale, steps)
    betas = np.linspace(-scale, scale, steps)
    losses = np.zeros((steps, steps))
    params = [model.gen[3].weight, model.gen[3].bias]
    orig_params = [p.clone() for p in params]
    for i, a in enumerate(tqdm(alphas, desc="Alpha")):
        for j, b in enumerate(betas):
            for p, o, d1v, d2v in zip(params, orig_params, d1, d2):
                p.data = o + a*d1v + b*d2v
            losses[i, j] = get_gan_loss(model, disc, dataloader, latent_dim)
    for p, o in zip(params, orig_params): p.data.copy_(o)
    return alphas, betas, losses

d1, d2 = get_random_directions(generator)
alphas, betas, losses = compute_loss_surface(generator, discriminator, subset_loader, d1, d2, GRID_STEPS, PERTURB_SCALE, LATENT_DIM)

X, Y = np.meshgrid(alphas, betas)
Z = losses

# 3D Surface Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
plt.colorbar(surf, label="Generator Loss")
ax.set_xlabel("α Direction"); ax.set_ylabel("β Direction"); ax.set_zlabel("Loss")
plt.title(f"GAN Loss Landscape ({GRID_STEPS}x{GRID_STEPS}) - 3D Surface")
plt.savefig(f"loss_landscape_3d_{GRID_STEPS}.png", dpi=300)
plt.close()

# 2D Heatmap
plt.figure(figsize=(8,6))
plt.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
plt.colorbar(label="Generator Loss")
plt.xlabel("α Direction"); plt.ylabel("β Direction")
plt.title(f"GAN Loss Landscape ({GRID_STEPS}x{GRID_STEPS}) - 2D Heatmap")
plt.savefig(f"loss_landscape_2d_{GRID_STEPS}.png", dpi=300)
plt.close()

#  1D SLICE PLOT
plt.figure(figsize=(8,6))
output_image_path_1d = "/kaggle/working/loss_landscape_1d_slice.png"
print(f"Saving 1D Slice Plot to: {output_image_path_1d}")
center_index = GRID_STEPS // 2
loss_slice = Z[center_index, :]
alpha_values = alphas
plt.plot(alpha_values, loss_slice, 'b-', linewidth=2)
plt.plot(0, loss_slice[center_index], 'rx', markersize=10, markeredgewidth=2, label='Trained Weights')
plt.xlabel("Direction 1 (Perturbation $\\alpha$)")
plt.ylabel("Generator Loss ($L_G$)")
plt.title("GAN Generator Loss Landscape - 1D Slice ($\\beta \\approx 0$)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.ylim(bottom=np.min(loss_slice) * 0.95, top=np.max(loss_slice) * 1.05)
plt.savefig(output_image_path_1d, dpi=300)
plt.close()
gc.collect()

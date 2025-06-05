import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.convlstm3d import ConvLSTM3D
from models.esrgan import ESRGANGenerator
from models.wgan_gp import WGANCritic
from utils import VideoAnomalyDataset
import torchvision.transforms as T

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
])

# Dataset
train_dataset = VideoAnomalyDataset("dataset/training", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Models
temporal_gen = ConvLSTM3D().to(device)
sr_gen = ESRGANGenerator().to(device)
critic = WGANCritic().to(device)

# Optimizers
optimizer_g = optim.Adam(list(temporal_gen.parameters()) + list(sr_gen.parameters()), lr=1e-4)
optimizer_d = optim.Adam(critic.parameters(), lr=1e-4)

# Training Loop
for epoch in range(10):
    for i, batch in enumerate(train_loader):
        video_seq = batch.float().to(device)  # [B, T, C, H, W]
        video_seq = video_seq.permute(0, 2, 1, 3, 4)  # to [B, C, T, H, W]

        # Generator forward
        fake_frames = temporal_gen(video_seq)  # [B, C, T, H, W]
        last_frame = fake_frames[:, :, -1, :, :]  # ESRGAN on last frame
        sr_frame = sr_gen(last_frame)

        # Train Discriminator
        optimizer_d.zero_grad()
        real_frame = video_seq[:, :, -1, :, :]
        loss_d = -(critic(real_frame).mean() - critic(sr_frame.detach()).mean())
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        loss_g = -critic(sr_frame).mean()
        loss_g.backward()
        optimizer_g.step()

        print(f"Epoch {epoch}, Step {i}, Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")

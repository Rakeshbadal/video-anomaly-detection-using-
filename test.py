import torch
from torch.utils.data import DataLoader
from models.convlstm3d import ConvLSTM3D
from models.esrgan import ESRGANGenerator
from utils import VideoAnomalyDataset
import torchvision.transforms as T
import matplotlib.pyplot as plt

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
temporal_gen = ConvLSTM3D().to(device)
sr_gen = ESRGANGenerator().to(device)
temporal_gen.eval()
sr_gen.eval()

# Dataset
transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
])

test_dataset = VideoAnomalyDataset("dataset/testing", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Testing Loop
for i, batch in enumerate(test_loader):
    video_seq = batch.float().to(device)
    video_seq = video_seq.permute(0, 2, 1, 3, 4)

    with torch.no_grad():
        fake_seq = temporal_gen(video_seq)
        sr_frame = sr_gen(fake_seq[:, :, -1, :, :])

    # Save or visualize
    output_img = sr_frame.squeeze().permute(1, 2, 0).cpu().numpy()
    plt.imshow(output_img)
    plt.title(f"Predicted Frame {i}")
    plt.savefig(f"results/frame_{i}.png")

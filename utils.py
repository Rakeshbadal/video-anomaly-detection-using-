import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class VideoAnomalyDataset(Dataset):
    def __init__(self, root_dir, sequence_length=10, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.samples = self._load_sequences()

    def _load_sequences(self):
        all_frames = sorted([os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith('.jpg')])
        return [all_frames[i:i+self.sequence_length] for i in range(0, len(all_frames)-self.sequence_length)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames = self.samples[idx]
        sequence = []
        for frame_path in frames:
            img = Image.open(frame_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            sequence.append(img)
        return np.stack(sequence, axis=0)

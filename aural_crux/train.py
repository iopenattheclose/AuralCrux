import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd

#this class get items from dataset
#goes into the folder,finds file(along with label) and converts it into mel spectrogram
class AudioDataset(Dataset):
    def __init__(self, data_directory,metadata_file, split = "train", transform = None):
        super().__init__()
        self.data_directory = Path(data_directory)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        if split == "train":
            self.metadata = self.metadata[self.metadata['fold'] != 5]
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5]

        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls:idx for idx,cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(self.class_to_idx)

    def __len__(self):
        return len(self.metadata)


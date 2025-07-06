import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
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
    
    def __getitem__(self, index):
        row  = self.metadata.iloc[index]
        audio_path = self.data_directory / "audio" / row["filename"]

        waveform, sample_rate = torchaudio.load(audio_path)
        # shape of above is [channels, samples] = [2,44000]

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim= True) #channels, samples] = [2,44000] -> [1,44000]
        
        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        
        return spectrogram, row["label"]


def train():

    audio_files_dir = Path("aural_crux/artifacts/audio_files")

    train_transform = nn.Sequential(
                      T.MelSpectrogram(
                          sample_rate=44100,
                          n_fft=1024,
                          hop_length=512,
                          n_mels=128,
                          f_max=11025, 
                          f_min=0
                      ),
                      T.AmplitudeToDB(),
                      #these two are similar to droput in nn but for audio
                      T.FrequencyMasking(freq_mask_param=30),#randomly masks 30 frq bins to zero
                      T.TimeMasking(time_mask_param=80)#sets time ranges to 0 
    )

    val_transform = nn.Sequential(
                      T.MelSpectrogram(
                          sample_rate=44100,
                          n_fft=1024,
                          hop_length=512,
                          n_mels=128,
                          f_max=11025, 
                          f_min=0
                      ),
                      T.AmplitudeToDB(),
    )


    train_dataset = AudioDataset(data_directory=audio_files_dir,
                                 metadata_file="aural_crux/artifacts/meta.csv",split="train",transform=train_transform)
    
    val_dataset = AudioDataset(data_directory=audio_files_dir,
                                metadata_file="aural_crux/artifacts/meta.csv",split="val",transform=val_transform)
    
    print(f"Training Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}")



            

if __name__=="__main__":
    train()
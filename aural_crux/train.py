import torch
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
import pandas as pd
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
from model import AudioCNN
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm 
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os



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

def mixup_data(x, y):
    #x - features, y- labels
    #combines two audio files (0.3 from 1st and 0.7 from 2nd)
    lam = np.random.beta(0.2, 0.2)#blending percentage 

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # (0.7 * audio1) + (0.3 * audio2)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_directory = "aural_crux/artifacts/summary"
    if not os.path.exists(summary_directory):
        os.makedirs(summary_directory)
    log_dir = f'{summary_directory}/tensorboard_logs/run_{timestamp}'
    writer = SummaryWriter(log_dir)

    audio_files_dir = Path("aural_crux/artifacts")

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

    test_transform = nn.Sequential(
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
    
    test_dataset = AudioDataset(data_directory=audio_files_dir,
                                metadata_file="aural_crux/artifacts/meta.csv",split="test",transform=test_transform)
    
    # print(f"Training Samples: {len(train_dataset)}")
    # print(f"Testing Samples: {len(test_dataset)}")

    #dataloader - loops through dataset and feeds it through nn in batches
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle= True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle= False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device=device)

    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # [1,0,0,0,0] -> [0.9 , 0.025 , 0.025 , 0.025, 0.025]
    optimizer = optim.AdamW(model.parameters(),lr=0.0005, weight_decay=0.01)
    #scheduler - adjusts the lr during training so model can learn more effetively

    #revisit
    #OneCycleLR over rides the lr in the optimizer defined above
    #scheduler created its own learning rate curve that goes from very low value ~0 to max_lr and then back to ~0 
    scheduler = OneCycleLR(
            optimizer,
            max_lr=0.002,
            epochs=num_epochs,
            steps_per_epoch=len(train_dataloader),
            pct_start=0.1 #10% of training inc lr and 90% of training dec lr   
    )

    best_accuracy = 0.0
    
    print("Start training.......")

    for epoch in range(num_epochs):
        model.train()

        #accumulated loss calculated batch by batch (cannot calculate loss for entire epoch at once due to OOM)
        epoch_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

        #data is spectrogram and target is the label 
        for data,target in progress_bar:
            data,target = data.to(device), target.to(device)

            #data mixing - ie data augmentation
            #perform mixing only for 30% of all cases
            if np.random.random() > 0.7:
                data, target_a, target_b, lam = mixup_data(data, target)
                output = model(data)
                loss = mixup_criterion(
                    criterion, output, target_a, target_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})


        avg_epoch_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
        writer.add_scalar(
            'Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Validation after each epoch
        model.eval()

        correct = 0
        total = 0
        val_loss = 0

        #no grad tells pytorch not to touch model weights
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(test_dataloader)

        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)

        print(
            f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy

        
        #add to utils.py
        model_directory = "aural_crux/artifacts/models"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        torch.save({
            'model_state_dict': model.state_dict(),
            'accuracy': accuracy,
            'epoch': epoch,
            'classes': train_dataset.classes
        }, f'{model_directory}/best_model.pth')
        print(f'New best model saved: {accuracy:.2f}%')

    writer.close()
    print(f'Training completed! Best accuracy: {best_accuracy:.2f}%')


            

if __name__=="__main__":
    train() 
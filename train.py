import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from models.voice_converter import BasicVoiceConverter
from utils.audio_processing import AudioProcessor

class VoiceDataset(Dataset):
    """Simple dataset for voice conversion"""
    
    def __init__(self, source_dir, target_dir, processor):
        self.source_files = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
        self.target_files = [f for f in os.listdir(target_dir) if f.endswith('.wav')]
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.processor = processor
        
    def __len__(self):
        return min(len(self.source_files), len(self.target_files))
    
    def __getitem__(self, idx):
        # Load source and target audio
        source_path = os.path.join(self.source_dir, self.source_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])
        
        source_audio = self.processor.load_audio(source_path)
        target_audio = self.processor.load_audio(target_path)
        
        source_mel = self.processor.audio_to_mel(source_audio)
        target_mel = self.processor.audio_to_mel(target_audio)
        
        return torch.FloatTensor(source_mel), torch.FloatTensor(target_mel)

def train_model():
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and optimizer
    model = BasicVoiceConverter().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Create dataset and dataloader
    processor = AudioProcessor()
    dataset = VoiceDataset('data/source_voices', 'data/target_voices', processor)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Training loop
    num_epochs = 100
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, (source_mel, target_mel) in enumerate(dataloader):
            source_mel = source_mel.to(device)
            target_mel = target_mel.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            converted_mel = model(source_mel, target_mel)
            
            # Calculate loss (convert source to target)
            loss = criterion(converted_mel, target_mel)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
        
        # Save model checkpoint
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f'models/voice_converter_epoch_{epoch}.pth')
    
    print("Training completed!")
    torch.save(model.state_dict(), 'models/voice_converter_final.pth')

if __name__ == "__main__":
    train_model()
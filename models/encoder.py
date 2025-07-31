import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentEncoder(nn.Module):
    """Extracts content features (what is being said)"""
    
    def __init__(self, input_dim=80, hidden_dim=256, latent_dim=128):
        super(ContentEncoder, self).__init__()
        
        # Convolutional layers for local feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        
        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            256, hidden_dim // 2, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, mel_spec):
        # mel_spec shape: (batch_size, n_mels, time_steps)
        
        # Convolutional processing
        x = self.conv_layers(mel_spec)  # (batch, 256, time_steps)
        
        # Transpose for LSTM (batch, time_steps, features)
        x = x.transpose(1, 2)
        
        # LSTM processing
        x, _ = self.lstm(x)  # (batch, time_steps, hidden_dim)
        
        # Project to latent space
        content_features = self.output_proj(x)  # (batch, time_steps, latent_dim)
        
        return content_features

class SpeakerEncoder(nn.Module):
    """Extracts speaker characteristics (who is speaking)"""
    
    def __init__(self, input_dim=80, speaker_dim=64):
        super(SpeakerEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Global average pooling to get fixed-size representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Speaker embedding
        self.speaker_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, speaker_dim)
        )
        
    def forward(self, mel_spec):
        # mel_spec shape: (batch_size, n_mels, time_steps)
        
        # Convolutional processing
        x = self.conv_layers(mel_spec)  # (batch, 512, time_steps)
        
        # Global pooling to get speaker representation
        x = self.global_pool(x)  # (batch, 512, 1)
        x = x.squeeze(2)  # (batch, 512)
        
        # Project to speaker embedding
        speaker_embedding = self.speaker_proj(x)  # (batch, speaker_dim)
        
        return speaker_embedding
import torch
import torch.nn as nn

class VoiceDecoder(nn.Module):
    """Combines content and speaker features to generate voice"""
    
    def __init__(self, content_dim=128, speaker_dim=64, hidden_dim=256, output_dim=80):
        super(VoiceDecoder, self).__init__()
        
        self.content_dim = content_dim
        self.speaker_dim = speaker_dim
        
        # Combine content and speaker features
        self.input_proj = nn.Linear(content_dim + speaker_dim, hidden_dim)
        
        # LSTM layers for sequence generation
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, content_features, speaker_embedding):
        # content_features: (batch, time_steps, content_dim)
        # speaker_embedding: (batch, speaker_dim)
        
        batch_size, time_steps, _ = content_features.shape
        
        # Expand speaker embedding to match time dimension
        speaker_expanded = speaker_embedding.unsqueeze(1).expand(batch_size, time_steps, -1)
        
        # Concatenate content and speaker features
        combined = torch.cat([content_features, speaker_expanded], dim=2)
        
        # Project to hidden dimension
        x = self.input_proj(combined)
        
        # LSTM processing
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Generate output mel-spectrogram
        output_mel = self.output_layers(x)
        
        return output_mel
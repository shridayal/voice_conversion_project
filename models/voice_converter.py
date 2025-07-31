import torch
import torch.nn as nn
from .encoder import ContentEncoder, SpeakerEncoder
from .decoder import VoiceDecoder

class BasicVoiceConverter(nn.Module):
    """Complete voice conversion model"""
    
    def __init__(self, mel_dim=80, content_dim=128, speaker_dim=64, hidden_dim=256):
        super(BasicVoiceConverter, self).__init__()
        
        # Initialize components
        self.content_encoder = ContentEncoder(mel_dim, hidden_dim, content_dim)
        self.speaker_encoder = SpeakerEncoder(mel_dim, speaker_dim)
        self.decoder = VoiceDecoder(content_dim, speaker_dim, hidden_dim, mel_dim)
        
    def forward(self, source_mel, target_mel):
        """
        source_mel: mel-spectrogram of source voice
        target_mel: mel-spectrogram of target voice (for speaker characteristics)
        """
        
        # Extract content from source
        content_features = self.content_encoder(source_mel)
        
        # Extract speaker characteristics from target
        speaker_embedding = self.speaker_encoder(target_mel)
        
        # Generate converted voice
        converted_mel = self.decoder(content_features, speaker_embedding)
        
        return converted_mel
    
    def convert_voice(self, source_mel, target_speaker_embedding):
        """Convert voice using pre-computed speaker embedding"""
        with torch.no_grad():
            content_features = self.content_encoder(source_mel)
            converted_mel = self.decoder(content_features, target_speaker_embedding)
        return converted_mel
    
    def extract_speaker_embedding(self, mel_spec):
        """Extract speaker embedding from mel-spectrogram"""
        with torch.no_grad():
            speaker_embedding = self.speaker_encoder(mel_spec)
        return speaker_embedding

# Test the model
if __name__ == "__main__":
    # Create model
    model = BasicVoiceConverter()
    
    # Test with dummy data
    batch_size, mel_dim, time_steps = 2, 80, 100
    source_mel = torch.randn(batch_size, mel_dim, time_steps)
    target_mel = torch.randn(batch_size, mel_dim, time_steps)
    
    # Forward pass
    converted = model(source_mel, target_mel)
    
    print(f"Input shape: {source_mel.shape}")
    print(f"Output shape: {converted.shape}")
    print("✅ Model created successfully!")
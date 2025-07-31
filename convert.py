import torch
import numpy as np
from models.voice_converter import BasicVoiceConverter
from utils.audio_processing import AudioProcessor

def convert_voice(source_audio_path, target_audio_path, output_path, model_path):
    """Convert source voice to target voice"""
    
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AudioProcessor()
    
    # Load model
    model = BasicVoiceConverter().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Process audio
    source_audio = processor.load_audio(source_audio_path)
    target_audio = processor.load_audio(target_audio_path)
    
    source_mel = processor.audio_to_mel(source_audio)
    target_mel = processor.audio_to_mel(target_audio)
    
    # Convert to tensors
    source_mel = torch.FloatTensor(source_mel).unsqueeze(0).to(device)
    target_mel = torch.FloatTensor(target_mel).unsqueeze(0).to(device)
    
    # Convert voice
    with torch.no_grad():
        converted_mel = model(source_mel, target_mel)
    
    # Convert back to audio
    converted_mel_np = converted_mel.squeeze(0).cpu().numpy()
    converted_audio = processor.mel_to_audio(converted_mel_np)
    
    # Save result
    processor.save_audio(converted_audio, output_path)
    print(f"✅ Voice converted! Output saved to: {output_path}")

if __name__ == "__main__":
    convert_voice(
        source_audio_path="data/source_voices/my_voice.wav",
        target_audio_path="data/target_voices/target_voice.wav", 
        output_path="data/output/converted_voice.wav",
        model_path="models/voice_converter_final.pth"
    )
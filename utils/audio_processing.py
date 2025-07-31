import librosa
import numpy as np
import soundfile as sf
import torch

class AudioProcessor:
    def __init__(self, sample_rate=22050, n_mels=80, hop_length=256):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = hop_length * 4
        self.n_fft = 1024
    
    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def audio_to_mel(self, audio):
        """Convert audio to mel-spectrogram"""
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [-1, 1]
        log_mel = (log_mel + 80) / 80
        
        return log_mel
    
    def mel_to_audio(self, mel_spec):
        """Convert mel-spectrogram back to audio (using Griffin-Lim)"""
        # Denormalize
        mel_spec = mel_spec * 80 - 80
        
        # Convert back to linear scale
        mel_spec = librosa.db_to_power(mel_spec)
        
        # Reconstruct audio using Griffin-Lim algorithm
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft
        )
        
        return audio
    
    def save_audio(self, audio, output_path):
        """Save audio to file"""
        sf.write(output_path, audio, self.sample_rate)

# Test the processor
if __name__ == "__main__":
    processor = AudioProcessor()
    
    # Test with a sample audio file
    audio = processor.load_audio("test_audio.wav")
    mel = processor.audio_to_mel(audio)
    reconstructed = processor.mel_to_audio(mel)
    processor.save_audio(reconstructed, "reconstructed.wav")
    
    print(f"Original audio shape: {audio.shape}")
    print(f"Mel-spectrogram shape: {mel.shape}")
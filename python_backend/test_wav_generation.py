import base64
import numpy as np
import soundfile as sf
import io
import math

# Test the WAV generation that Flutter will now use
print('Testing new WAV audio generation...')

# Simulate the new WAV generation from Flutter
sample_rate = 16000
duration_seconds = 3
total_samples = sample_rate * duration_seconds

# Generate realistic audio waveform
audio_samples = []
for i in range(total_samples):
    time = i / sample_rate
    wave = 0.0
    
    # Add multiple frequency components
    wave += 0.3 * math.sin(2 * math.pi * 440 * time)  # 440 Hz
    wave += 0.2 * math.sin(2 * math.pi * 880 * time)  # 880 Hz
    wave += 0.1 * math.sin(2 * math.pi * 220 * time)  # 220 Hz
    
    # Convert to 16-bit PCM
    sample = int(wave * 32767)
    sample = max(-32768, min(32767, sample))
    audio_samples.append(sample & 0xFF)        # Low byte
    audio_samples.append((sample >> 8) & 0xFF) # High byte

# Create WAV header
def create_wav_header(data_size, sample_rate, channels):
    byte_rate = sample_rate * channels * 2
    file_size = 36 + data_size
    
    header = bytearray()
    header.extend(b'RIFF')
    header.extend(file_size.to_bytes(4, 'little'))
    header.extend(b'WAVE')
    header.extend(b'fmt ')
    header.extend((16).to_bytes(4, 'little'))
    header.extend((1).to_bytes(2, 'little'))  # PCM
    header.extend(channels.to_bytes(2, 'little'))
    header.extend(sample_rate.to_bytes(4, 'little'))
    header.extend(byte_rate.to_bytes(4, 'little'))
    header.extend((channels * 2).to_bytes(2, 'little'))
    header.extend((16).to_bytes(2, 'little'))  # bits per sample
    header.extend(b'data')
    header.extend(data_size.to_bytes(4, 'little'))
    
    return header

# Create complete WAV file
wav_header = create_wav_header(len(audio_samples), sample_rate, 1)
wav_data = bytes(wav_header) + bytes(audio_samples)
audio_base64 = base64.b64encode(wav_data).decode()

print(f'Generated WAV data: {len(wav_data)} bytes')
print(f'Audio base64 length: {len(audio_base64)}')

try:
    # Test if it can be decoded properly
    audio_bytes = base64.b64decode(audio_base64)
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    print(f'SUCCESS: Audio shape: {audio.shape}, Sample rate: {sr}')
    print(f'Audio stats - Min: {audio.min():.3f}, Max: {audio.max():.3f}, Mean: {audio.mean():.3f}')
    print('Valid audio data generated! This will work with the ML pipeline.')
    
except Exception as e:
    print(f'ERROR: {e}')

import wave
import math
import struct
import os

def generate_beep(filename, duration_sec, freq_hz, vol=0.5):
    sample_rate = 44100
    n_samples = int(sample_rate * duration_sec)
    
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        
        for i in range(n_samples):
            # Sine wave
            value = int(32767.0 * vol * math.sin(2.0 * math.pi * freq_hz * i / sample_rate))
            data = struct.pack('<h', value)
            f.writeframes(data)
    print(f"Generated {filename}")

def main():
    output_dir = os.path.expanduser("~/DROWSINESS_DETECTION/audio_alerts")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate different tones for different alerts
    alerts = {
        'soft_warning.wav': (0.5, 600),   # Short, medium pitch
        'mild_alert.wav': (0.8, 800),     # Longer, higher pitch
        'strong_alert.wav': (1.0, 1000),  # Long, high pitch
        'urgent_alert.wav': (1.5, 1200)   # Very long, very high pitch (siren-like)
    }
    
    for name, (dur, freq) in alerts.items():
        path = os.path.join(output_dir, name)
        generate_beep(path, dur, freq)

if __name__ == "__main__":
    main()

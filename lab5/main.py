import numpy as np
import soundfile as sf
from numpy.fft import fft, ifft
from scipy.signal import get_window
import matplotlib.pyplot as plt

def noise_reduction(
        input_file, 
        output_file, 
        frame_size=4096, 
        overlap=0.5, 
        noise_start=0, 
        noise_end=3000, 
        suppression_factor=3, 
        protection_factor=0.0001
        ):
    
    audio, sample_rate = sf.read(input_file)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    audio /= peak

    hop_size = int(frame_size * (1 - overlap))
    window = get_window('hann', frame_size)

    noise_frames = []
    for i in range(noise_start * sample_rate // 1000, min(noise_end * sample_rate // 1000, len(audio) - frame_size), hop_size):
        frame = audio[i:i + frame_size] * window
        noise_fft = fft(frame)
        noise_frames.append(np.abs(noise_fft))

    noise_profile = np.mean(noise_frames, axis=0) if noise_frames else np.zeros(frame_size)

    output_signal = np.zeros(len(audio))
    window_sum = np.zeros(len(audio))

    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size] * window
        frame_fft = fft(frame)

        magnitude = np.abs(frame_fft)
        phase = np.angle(frame_fft)

        clean_magnitude = np.maximum(magnitude - suppression_factor * noise_profile, protection_factor * magnitude)

        # Тригонометрическая форма комплексного числа: z = r * exp(i*arg(z))
        clean_fft = clean_magnitude * np.exp(1j * phase)
        clean_frame = np.real(ifft(clean_fft))

        output_signal[i:i + frame_size] += clean_frame
        window_sum[i:i + frame_size] += window

    window_sum[window_sum == 0] = 1
    output_signal /= window_sum

    output_signal = output_signal * 32767
    output_signal = np.clip(output_signal, -32768, 32767)
    output_signal = output_signal.astype(np.int16)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.specgram(audio, Fs=sample_rate, NFFT=frame_size, noverlap=hop_size)
    plt.colorbar()
    plt.title('Original Audio Spectrogram')
    plt.ylabel('Frequency (Hz)')

    plt.subplot(2, 1, 2)
    plt.specgram(output_signal, Fs=sample_rate, NFFT=frame_size, noverlap=hop_size)
    plt.colorbar()
    plt.title('Cleaned Audio Spectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')

    plt.tight_layout()
    plt.show()

    sf.write(output_file, output_signal, sample_rate)


noise_reduction('crowd.wav', 'output.wav')
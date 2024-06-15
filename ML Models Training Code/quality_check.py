import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_waveforms_and_spectrogram(original_file, processed_file):
    # Load the audio files
    original_signal, sr_original = librosa.load(original_file, sr=None)
    processed_signal, sr_processed = librosa.load(processed_file, sr=None)

    # Calculate the difference signal
    difference_signal = original_signal[:len(processed_signal)] - processed_signal

    # Create subplots
    fig, ax = plt.subplots(4, 1, figsize=(12, 14))

    # Plot original audio
    ax[0].set_title('Original Audio')
    librosa.display.waveshow(original_signal, sr=sr_original, ax=ax[0])
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')

    # Plot processed audio
    ax[1].set_title('Processed Audio')
    librosa.display.waveshow(processed_signal, sr=sr_processed, ax=ax[1])
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Amplitude')

    # Plot difference audio
    ax[2].set_title('Difference Audio')
    librosa.display.waveshow(difference_signal, sr=sr_original, ax=ax[2], color='g')
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Amplitude')

    D_original = librosa.amplitude_to_db(np.abs(librosa.stft(original_signal)), ref=np.max)
    D_processed = librosa.amplitude_to_db(np.abs(librosa.stft(processed_signal)), ref=np.max)

    # Ensure the spectrograms are the same size
    shape = min(D_original.shape[1], D_processed.shape[1])
    D_original = D_original[:, :shape]
    D_processed = D_processed[:, :shape]

    D_diff = D_original - D_processed
    # Calculate the time axis for the spectrogram (assuming hop_length is 512, adjust as needed)
    hop_length = 512
    time_bins = np.arange(D_diff.shape[1]) * hop_length / float(sr_original)
    # Calculate the frequency axis for the spectrogram
    freq_bins = librosa.fft_frequencies(sr=sr_original, n_fft=D_diff.shape[0])

    img = ax[3].imshow(D_diff, aspect='auto', origin='lower',
                       extent=[time_bins[0], time_bins[-1], freq_bins[0], freq_bins[-1]], cmap='coolwarm')
    ax[3].set_title('Spectrogram Difference')
    ax[3].set_xlabel('Time (s)')
    ax[3].set_ylabel('Frequency (Hz)')
    fig.colorbar(img, ax=ax[3], format='%+2.0f dB')

    ax[3].set_title('Spectrogram Difference')
    ax[3].set_xlabel('Time')
    ax[3].set_ylabel('Frequency')
    fig.colorbar(img, ax=ax[3], format='%+2.0f dB')

    plt.tight_layout()
    plt.show()


def main():
    # Replace these with the actual paths to your files
    original_file = 'E:\Final Year\Final Year Project\pythonProject\Dataset\genres_original\jazz\jazz.00007.wav'
    processed_file = 'E:\Final Year\Final Year Project\pythonProject\Processed-Dataset\jazz\jazz.00007.wav'

    plot_waveforms_and_spectrogram(original_file, processed_file)


if __name__ == "__main__":
    main()

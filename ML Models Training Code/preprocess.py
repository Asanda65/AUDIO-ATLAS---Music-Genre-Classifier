import librosa
import soundfile as sf
import os

# Path to the 'genre original' directory
genre_directory = r"E:\pythonProject\Dataset\genres_original"

# Directory to save processed files
processed_directory = r"E:\pythonProject\Processed-Dataset"

# Create the processed directory if it doesn't exist
if not os.path.exists(processed_directory):
    os.makedirs(processed_directory)

# Counter for processed files
processed_count = 0

# List to keep track of corrupt files
corrupt_files = []

# Iterate over each genre subfolder
for genre_subfolder in os.listdir(genre_directory):
    genre_subfolder_path = os.path.join(genre_directory, genre_subfolder)
    processed_subfolder_path = os.path.join(processed_directory, genre_subfolder)

    # Create a subfolder in processed directory for each genre
    if not os.path.exists(processed_subfolder_path):
        os.makedirs(processed_subfolder_path)

    # Check if it's a directory
    if os.path.isdir(genre_subfolder_path):
        # Process each file in the genre subfolder
        for file in os.listdir(genre_subfolder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(genre_subfolder_path, file)
                try:
                    # Load the audio file
                    signal, sr = librosa.load(file_path, sr=None)

                    # Normalize the signal
                    signal = librosa.util.normalize(signal)

                    # Optional: Trim silence
                    signal, _ = librosa.effects.trim(signal)

                    # Save the processed signal
                    processed_file_path = os.path.join(processed_subfolder_path, file)
                    sf.write(processed_file_path, signal, sr)

                    processed_count += 1

                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    corrupt_files.append(file_path)

    # Print update after each genre
    print(f"Completed processing {processed_count} files in genre {genre_subfolder}")

# Handle corrupt files by deleting them
for corrupt_file in corrupt_files:
    os.remove(corrupt_file)
    print(f"Deleted corrupt file: {corrupt_file}")

# Final update
print(f"Total files processed: {processed_count}")
print(f"Corrupt files detected and deleted: {len(corrupt_files)}")

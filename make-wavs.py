import os
import ffmpeg

# Input audio file
import argparse

parser = argparse.ArgumentParser(description="Process some audio files.")
parser.add_argument('directory', type=str,
                    help='Input directory containing audio files')
parser.add_argument('output_directory', type=str,
                    help='Output directory to store processed audio files')
parser.add_argument('--engine', type=str, default="dtw-ra",
                    help='Engine name (default: dtw-ra)')

args = parser.parse_args()

directory = args.directory
output_directory = os.path.join(args.output_directory, args.engine)
engine = args.engine

def convert_to_wav(input_file, output_file):
    """
    Converts an audio file to WAV format using ffmpeg with specified parameters.
    - Sample rate: 16000 Hz
    - Audio channels: 1 (mono)
    - Audio codec: pcm_s16le
    """

    print(f"{input_file} to {output_file}")
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_file, ar=16000, ac=1, c='pcm_s16le')
            .run(overwrite_output=True)
        )
        print(
            f"Converted {input_file} to {output_file} with format -ar 16000 -ac 1 -c:a pcm_s16le")
    except Exception as e:
        print(f"Error converting file: {e}")

# iterate over all files in the directory
for filename in os.listdir(directory):
    # split the filename into name and extension
    name, extension = os.path.splitext(filename)


    if (
        extension.lower() == '.mp3'
        and not os.path.exists(os.path.join(directory, f'{name}.wav'))
    ):
        
        mp3_file = os.path.join(directory, filename)   
        wav_file = os.path.join(directory, f'{name}.wav')

        # Convert the input audio file to WAV format
        if not os.path.exists(wav_file):
            print(f"WAV does not exist for {name} at {wav_file}")
            convert_to_wav(mp3_file, wav_file)
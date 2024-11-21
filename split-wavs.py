from concurrent.futures import ProcessPoolExecutor
import os
import json
import time
from pydub import AudioSegment
import pandas as pd
import ffmpeg
import whisper
from faster_whisper import WhisperModel
import os
import re
import jiwer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from tqdm import tqdm, trange

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# whisper_model = whisper.load_model('large-v2')
whisper_model = WhisperModel('large-v2', device="cuda", compute_type="float16")
whisper_norm = BasicTextNormalizer()

import argparse

parser = argparse.ArgumentParser(description='Process some audio files and evaluate using Whisper model.')
parser.add_argument('--directory', required=True, type=str,
                    help='The directory containing the evaluation dataset')
parser.add_argument('--input-directory', required=True, type=str,
                    help='The input directory where processed audio files are stored')
parser.add_argument('--output-directory', required=True, type=str,
                    help='The output directory where results will be saved')
parser.add_argument('--engine', default="dtw-ra", type=str,
                    help='The engine used for processing (default: dtw-ra)')

args = parser.parse_args()

directory = args.directory
engine = args.engine
input_directory = args.input_directory
output_directory = args.output_directory

os.makedirs(output_directory, exist_ok=True)

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


def process_audio_segments(wav_file, timestamps, output_dir, max_duration=30000):
    """
    Processes the audio segments based on timestamps and saves them with increasing index filenames.
    Creates a CSV mapping the filenames to the corresponding sentences.
    """
    audio = AudioSegment.from_wav(wav_file)
    data = []
    total_duration = 0
    segment_index = 0
    total_sentence = []
    segment_start = timestamps[0][0]*1000
    segment_end = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_filename = os.path.join(output_dir, "segments.csv")
    if os.path.exists(csv_filename):
        return

    print(f"Processing {wav_file}")

    timestamps = list(filter(lambda item: (
        item[0] > 0 and item[1] >= item[0]), timestamps))

    print(' ', end='', flush=True)
    for index in trange(len(timestamps)):
        start, end, sentence = timestamps[index]

        start_ms = start * 1000
        end_ms = end * 1000

        next_start_ms = timestamps[index+1][0] * \
            1000 if index+1 < len(timestamps) else end_ms

        median_end_ms = ((next_start_ms - end_ms) // 2) + end_ms

        duration = median_end_ms - start_ms
        # print(f"Duration: {duration} {sentence}")

        if int(median_end_ms - segment_start) >= max_duration and segment_end != segment_start:
            export_segment(audio, segment_start, segment_end,
                           segment_index, total_sentence, data, output_dir)

            segment_index += 1
            total_duration = 0
            total_sentence = []
            segment_start = segment_end

        segment_end = median_end_ms

        if (duration > max_duration):
            total_duration = 0
            total_sentence = []
            segment_start = segment_end
        else:
            total_sentence.append(sentence)
            total_duration += duration

    if total_duration > 0:
        export_segment(audio, segment_start, segment_end,
                       segment_index, total_sentence, data, output_dir)

    df = pd.DataFrame(data, columns=["filename", "sentence", "duration"])
    df.to_csv(csv_filename, index=False)
    print(f"Segments and sentences saved to {csv_filename}")


def export_segment(audio, segment_start, segment_end, segment_index, total_sentence, data, output_dir):
    segment = audio[segment_start:segment_end]
    segment_filename = os.path.join(
        output_dir, f"{segment_index+1}.wav")
    segment.export(segment_filename, format="wav", parameters=[
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"])

    whisper_segments, whisper_info = whisper_model.transcribe(
        segment_filename, beam_size=5)
    # whisper_result = whisper_model.transcribe(segment_filename, decode_options={
    #     'language': 'nl'
    # })

    total_sentence = " ".join(total_sentence)

    # The transcription will actually run here.
    whisper_transcript = " ".join(
        map(lambda item: item.text, whisper_segments))

    reference_length = len(re.findall(r'\w+', total_sentence))
    whisper_length = len(re.findall(r'\w+', whisper_transcript))

    change_percent = (
        (whisper_length - reference_length)/reference_length) * 100

    jiwer_score = jiwer.process_words(
        whisper_norm(total_sentence), whisper_norm(whisper_transcript))

    # jiwer_char_score = jiwer.process_characters(whisper_norm(total_sentence), whisper_norm(whisper_transcript))

    # print(segment_index, jiwer.visualize_alignment(jiwer_score))
    # print(f"CER: {jiwer_char_score.cer}")
    # print(segment_index, jiwer.visualize_alignment(jiwer_char_score))

    # KEEP, old values for reference: if abs(change_percent) > 25 or jiwer_score.wer > 0.5:

    if abs(change_percent) > 20 or jiwer_score.wer > 0.3:
        pass
        # print("WHISP:" + str(whisper_length) + " " + whisper_transcript)
        # print("REF  :" + str(reference_length) + " " + total_sentence)
        # print(f"CHANGE AT #{segment_index}: " +
        #   str(int(change_percent)) + "%\n\n")
    else:

        # print(f"WER: {jiwer_score.wer}")
        # print(f"MER: {jiwer_score.mer}")
        # print(f"WIL: {jiwer_score.wil}")
        # print(f"WIP: {jiwer_score.wip}")
        # print(f"INS: {jiwer_score.insertions}")
        # print(f"SUB: {jiwer_score.substitutions}")
        # print(f"DEL: {jiwer_score.deletions}")
        # print(f"HTS: {jiwer_score.hits}")
        # print(f"CHANGE AT #{segment_index}: " +
        #       str(int(change_percent)) + "%")
        total_duration = segment_end - segment_start
        data.append([segment_filename, total_sentence, total_duration])


def process_file(filename):
    # split the filename into name and extension
    name, extension = os.path.splitext(filename)

    if (
        extension == '.json'
        and os.path.exists(os.path.join(directory, f'{name}.txt'))
        and os.path.exists(os.path.join(directory, f'{name}.wav'))
    ):

        mp3_file = os.path.join(directory, f'{name}.MP3')
        wav_file = os.path.join(directory, f'{name}.wav')
        json_file_path = os.path.join(
            input_directory, f"{name}.json")

        if not os.path.exists(json_file_path):
            print(f"JSON does not exist for {name}")
            return

        # Convert the input audio file to WAV format
        if not os.path.exists(wav_file):
            print(f"WAV does not exist for {name} at {wav_file}")
            convert_to_wav(mp3_file, wav_file)

        with open(json_file_path, 'r', encoding="utf8") as f:
            data = json.load(f)

            # List of timestamps (start, end, sentence)
            timestamps = [(sentence['startTime'], sentence['endTime'], sentence['text'])
                          for item in data
                          for sentence in item['timeline']]

            # Output directory for audio segments
            segments_output_dir = os.path.join(
                output_directory, f"audio_segments_{name}")

            # Process audio segments
            process_audio_segments(
                wav_file, timestamps, segments_output_dir)


def main():
    # Input audio file

    # Get list of files to process
    files_to_process = [file for file in os.listdir(input_directory)
                        if os.path.splitext(file)[1] == '.json'
                        and
                        os.path.exists(os.path.join(
                            directory, f"{os.path.splitext(file)[0]}.txt"))
                        and
                        os.path.exists(os.path.join(
                            directory, f"{os.path.splitext(file)[0]}.wav"))
                        ]

    # Use ProcessPoolExecutor to process files concurrently
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(process_file, files_to_process)


if __name__ == "__main__":
    main()

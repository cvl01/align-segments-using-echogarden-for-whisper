import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from faster_whisper import WhisperModel
import re
import jiwer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(
    description="Process audio segments with Whisper and evaluate using WER."
)
parser.add_argument(
    "--engine",
    type=str,
    default="dtw-ra",
    help="Engine name to use for directory structure",
)
parser.add_argument(
    "--dataset_output_directory",
    type=str,
    required=True,
    help="Directory where the dataset will be stored",
)
parser.add_argument(
    "--csv_path",
    type=str,
    required=True,
    help="Path to the CSV file containing merged segments",
)

args = parser.parse_args()

engine = args.engine
dataset_output_directory = os.path.join(args.dataset_output_directory, engine)

csv_path = args.csv_path

os.makedirs(dataset_output_directory, exist_ok=True)

whisper_model = WhisperModel("large-v2", device="cuda", compute_type="float16")
whisper_norm = BasicTextNormalizer()


def filter_by_wer(csv_file_path):
    # Load the CSV file
    data = pd.read_csv(csv_file_path)

    # Initialize a list to store rows that meet the criteria
    filtered_rows = []

    # Iterate over each row in the DataFrame
    for index, row in tqdm(data.iterrows()):
        audio_file = r"C:/Users/luik001c/Documents/echogarden/" + row["audio"]
        reference_sentence = row["sentence"]

        whisper_segments, whisper_info = whisper_model.transcribe(
            audio_file, beam_size=5
        )
        # whisper_result = whisper_model.transcribe(segment_filename, decode_options={
        #     'language': 'nl'
        # })

        # The transcription will actually run here.
        transcription = " ".join(map(lambda item: item.text, whisper_segments))

        reference_length = len(re.findall(r"\w+", reference_sentence))
        whisper_length = len(re.findall(r"\w+", transcription))

        # Calculate the WER
        change_percent = ((whisper_length - reference_length) / reference_length) * 100

        jiwer_score = jiwer.process_words(
            whisper_norm(reference_sentence), whisper_norm(transcription)
        )

        # Append the row to filtered_rows if WER is <= 30%
        if abs(change_percent) < 20 and jiwer_score.wer < 0.3:
            filtered_rows.append(
                {
                    "audio": row["audio"],
                    "sentence": reference_sentence,
                    "transcription": transcription,
                    "wer": jiwer_score.wer,
                    "change": change_percent,
                }
            )

    # Create a new DataFrame from the filtered rows
    filtered_data = pd.DataFrame(filtered_rows)

    # Save the filtered data to a new CSV file or return it
    filtered_data.to_csv("filtered_data.csv", index=False)

    return filtered_data


# Example usage
filtered_df = filter_by_wer(csv_path)

filtered_df["audio"] = (
    "/mnt/c/Users/luik001c/Documents/echogarden/" + filtered_df["audio"]
)


train_df, test_df = train_test_split(filtered_df, test_size=0.10, random_state=42)


filtered_df.to_csv(
    os.path.join(dataset_output_directory, "filtered_segments.csv"),
    index=False,
    quoting=csv.QUOTE_ALL,
)
train_df.to_csv(
    os.path.join(dataset_output_directory, "filtered_segments_train.csv"),
    index=False,
    quoting=csv.QUOTE_ALL,
)
test_df.to_csv(
    os.path.join(dataset_output_directory, "filtered_segments_test.csv"),
    index=False,
    quoting=csv.QUOTE_ALL,
)

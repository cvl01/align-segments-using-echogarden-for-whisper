from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import json
import random
import subprocess


import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(
    description="Process dataset files with specified engine."
)

# Add required arguments
parser.add_argument("directory", type=str, help="The directory containing the files.")
parser.add_argument(
    "--output_directory",
    type=str,
    required=True,
    help="The output directory for processed files.",
)

# Add optional argument with a default value
parser.add_argument(
    "--engine",
    type=str,
    choices=["dtw", "dtw-ra", "whisper"],
    default="dtw-ra",
    help="The engine to use for processing. Choices: dtw, dtw-ra, whisper (default: dtw-ra).",
)

# Parse the command-line arguments
args = parser.parse_args()

directory = args.directory
engine = args.engine
output_directory = os.path.join(args.output_directory, engine)

os.makedirs(output_directory, exist_ok=True)


def process_file(args):
    filename, worker_id = args

    # split the filename into name and extension
    name, extension = os.path.splitext(filename)

    output_json_path = os.path.join(output_directory, f"{name}.json")
    output_srt_path = os.path.join(output_directory, f"{name}.srt")

    if os.path.exists(output_json_path):
        return

    # check if we have both .wav and .txt for the same filename
    if extension == ".wav" and os.path.exists(os.path.join(directory, f"{name}.txt")):
        print(f"Processing {filename}...")

        # construct the command to run echogarden
        cmd = ["echogarden align"]

        # modify the command with the actual file names
        cmd.append('"' + os.path.join(directory, filename) + '"')
        cmd.append('"' + os.path.join(directory, f"{name}.txt") + '"')

        cmd.append('"' + output_srt_path + '"')
        cmd.append('"' + output_json_path + '"')

        cmd.append("--language=nl-NL")
        cmd.append("--crop=true")
        cmd.append(f"--engine={engine}")

        gpu_id = worker_id % 2  # Assign GPU ID 0 or 1 based on worker ID
        if engine == "dtw-ra":
            cmd.append("--dtw.phoneAlignmentMethod=interpolation")
            cmd.append("--recognition.engine=whisper.cpp")
            cmd.append("--recognition.whisperCpp.model=large-v2")
            cmd.append("--recognition.whisperCpp.build=cublas-12.4.0")
            cmd.append("--recognition.whisperCpp.enableGPU=true")
            cmd.append("--recognition.whisperCpp.repetitionThreshold=2")
        if engine == "whisper":
            cmd.append("--whisper.model=small")
            cmd.append("--whisper.encoderProvider=cpu")
            cmd.append("--whisper.decoderProvider=cpu")

        cmd = " ".join(cmd)
        cmd = f"set CUDA_VISIBLE_DEVICES={
            gpu_id} & set DML_VISIBLE_DEVICES={gpu_id} & {cmd}"

        print(f"{cmd}")

        # run the command using subprocess
        try:
            subprocess.run(
                cmd, shell=True, stdout=subprocess.DEVNULL, capture_output=False
            )
        except:
            print("FAIL!")


def initializer(counter):
    global process_counter
    process_counter = counter


def worker_id():
    global process_counter
    return process_counter.get_and_increment()


class Counter:
    def __init__(self, initial=0):
        self.value = multiprocessing.Value("i", initial)
        self.lock = multiprocessing.Lock()

    def get_and_increment(self):
        with self.lock:
            current_value = self.value.value
            self.value.value += 1
        return current_value


if __name__ == "__main__":

    os.makedirs(output_directory, exist_ok=True)

    # Get list of files to process
    files_to_process = [
        file
        for file in os.listdir(directory)
        if os.path.splitext(file)[1] == ".wav"
        and os.path.exists(os.path.join(directory, f"{os.path.splitext(file)[0]}.txt"))
    ]

    random.shuffle(files_to_process)

    # Initialize the counter
    process_counter = Counter()

    # Use ProcessPoolExecutor to process files concurrently
    with ProcessPoolExecutor(
        max_workers=2, initializer=initializer, initargs=(process_counter,)
    ) as executor:
        executor.map(process_file, [(file, worker_id()) for file in files_to_process])

    print("Done!")

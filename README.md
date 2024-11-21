# Split long audio + transcript into 30-second training chunks for Whisper


## Usage steps

1. Install echogarden (globally on system, is run via cmd)

2. `align.py` generates the 30-second audio fragments. It creates a folder per input file and puts the 30s clips into this folder, together with `segments.csv`

3. `split-wavs.py` creates wav files from the timeline generated in the align step.

4. `combine.py` combines the chunks from multiple input folders into one big csv file. Also splits in train and test portions. 

5. `filter-segments.py` filters out segments of which the predicted sentence has a large WER compared to the reference. This is done to catch mis-aligned segments. 

`make-wavs.py` and `fix-encoding-txt.py` are helper scripts. 
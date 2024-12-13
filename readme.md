# Human-in-the-Loop Beat Tracking

## Setup

```bash
pip install -r requirements.txt
```

## Data

The data used in this project is from the MAPS database, which is a collection of classical piano music.

Data is stored in the `data` directory, including the audio files and the corresponding annotations.

### Read the data annotation

```bash
python save_annotations.py
```

## Usage


### Use MediaPipe to process a video

```bash
python process_video.py --input_video_path /path/to/video.mp4 --output_video_path /path/to/output_video.mp4
```

The processed video will be saved to the output path.
The raw gesture data will be saved to a JSON file in the same directory as the output video.


### Process gesture data to extract features

```bash
python process_data.py --gesture_data_path /path/to/gesture_data.json
```

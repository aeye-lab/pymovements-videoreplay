# pymovements-videoreplay

`pymovements-videoreplay` is an extension to the `pymovements` library designed for visualizing and analyzing
eye-tracking data through synchronized video replays. This tool enables researchers to replay participant eye movements
on visual stimuli, providing both raw and event-based overlays for detailed analysis.

## Features

- **Pure Video Replay**: Overlay eye-tracking data on visual stimuli, with options to toggle between raw and event-based
  data.
- **Playback Speed Control**: Real-time and slow-motion replay for detailed visualization of rapid eye movements.
- **Export Annotated Videos**: Save replays with gaze overlays as video files (e.g., MP4 or GIF) for presentations and
  reports.
- **Fixation-Based Navigation**: Interactive navigation through fixations, with context-aware visualization of past and
  future fixations.
- **Public Dataset Integration**: Load and explore eye-tracking data from public datasets.

## Installation

Install the package using pip:

```bash
pip install pymovements-videoreplay
```

Or, if you're developing the package locally:

```bash
pip install -e .
```

## Usage

### Basic Replay Example

```python
from videoreplay import VideoPlayer

# Initialize the video player
player = VideoPlayer(stimulus_path="video.mp4", gaze_data="path/to/gaze_data.csv")

# Play the video with gaze overlay
player.play()
```

### Export Replay

```python
player.export(output_path="output_video.mp4")
```

### Load Public Dataset

```python
import pymovements as pm

dataset = pm.Dataset('ToyDataset', path='data/ToyDataset')
dataset.download()
dataset.load()
```

## Dependencies

- `pymovements`
- `matplotlib`
- `numpy`
- `opencv-python`
- `pandas`
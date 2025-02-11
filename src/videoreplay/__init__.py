"""
videoreplay: A package for replaying eye-tracking recordings with gaze overlays.

This package provides tools for visualizing and analyzing eye-tracking data,
including video playback with gaze overlays and integration with public datasets.

Main Components:
- `VideoPlayer`: For replaying videos with gaze overlays.

Example Usage:
    from videoreplay import VideoPlayer

    player = VideoPlayer(video_path="stimulus.mp4",
                     dataset_path="data/ToyDataset",
                     dataset_name="ToyDataset")
"""

from .video_player import VideoPlayer

__all__ = ["VideoPlayer"]

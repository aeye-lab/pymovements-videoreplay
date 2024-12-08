"""
videoreplay: A package for replaying eye-tracking recordings with gaze overlays.

This package provides tools for visualizing and analyzing eye-tracking data,
including video playback with gaze overlays and integration with public datasets.

Main Components:
- `VideoPlayer`: For replaying videos with gaze overlays.

Example Usage:
    from videoreplay import VideoPlayer

    # Initialize videoplayer
    player = VideoPlayer(stimulus_path="video.mp4", gaze_data="path/to/gaze_data.csv")
    player.play()
"""

from .video_player import VideoPlayer

__all__ = ["VideoPlayer"]

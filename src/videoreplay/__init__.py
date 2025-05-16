"""
videoreplay
===========

Tools for visualising and analysing eye-tracking data, including video
or image playback with gaze overlays and helpers for column mapping.

Main Components
---------------
- **VideoPlayer** – plays a stimulus (image/video) with one-or-many gaze
  recordings overlaid, and can export the replay.
- **ColumnMappingDialog** – a Tk-inter dialog that lets the user map the
  column names in their CSV file to the fields `VideoPlayer` expects.
"""

from .video_player import VideoPlayer
from .column_mapping_dialog import ColumnMappingDialog

__all__ = ["VideoPlayer", "ColumnMappingDialog"]

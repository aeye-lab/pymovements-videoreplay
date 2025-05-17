"""Tools for visualising and analysing eye-tracking data.

Includes image or video playback with gaze overlays
and small helper dialogs.

Main Components
---------------

- **VideoPlayer** – plays a stimulus (image/video) with one-or-many gaze
  recordings overlaid and can export the replay.
- **ColumnMappingDialog** – Tk-inter dialog for mapping CSV column names
  to the fields `VideoPlayer` expects.
- **SessionSelectDialog** – Tk-inter dialog for choosing one of the
  loaded recording sessions before fixation navigation.
"""
from .column_mapping_dialog import ColumnMappingDialog
from .session_select_dialog import SessionSelectDialog
from .video_player import VideoPlayer

__all__: list[str] = [
    'VideoPlayer',
    'ColumnMappingDialog',
    'SessionSelectDialog',
]

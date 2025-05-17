"""Replay eye-tracking data on image or video stimuli with gaze overlay.

This module provides the `VideoPlayer` class to visualize and export
gaze replays from fixation data on both image and video stimuli.
"""
from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from videoreplay.column_mapping_dialog import ColumnMappingDialog
from videoreplay.session_select_dialog import SessionSelectDialog
# ── standard library ────────────────────────────────────────────────
# ── third-party libraries ───────────────────────────────────────────
# ── local package ───────────────────────────────────────────────────


class VideoPlayer:
    """Handles replay of eye-tracking data on stimuli (image or video).

    Parameters
    ----------
    stimulus_path : str
        Path to the stimulus image or video file.
    dataset_path : str
        Path to the directory containing the eye-tracking CSV data.
    recording_sessions : list[str]
        List of recording session labels used to filter the dataset.
    """

    def __init__(self, stimulus_path: str, dataset_path: str, recording_sessions: list[str]):
        self.stimulus_path = stimulus_path
        normalized_stimulus_name = self._normalize_stimulus_name(stimulus_path)
        self.is_image = self._is_image()

        self.overlay_colors = [
            (255, 0, 0),  # Blue
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (0, 165, 255),  # Orange
            (255, 255, 0),  # Cyan
        ]  # Color are in BGR format and not RGB
        self.dot_radius = 5
        self.gaze_dfs: list[tuple[str, pd.DataFrame]] = []

        root = tk.Tk()
        root.withdraw()
        mapping = ColumnMappingDialog(root, title='Column Mapping').result
        root.destroy()

        if mapping is None:
            raise ValueError('Column mapping configuration cancelled by user.')

        column_mapping = {
            mapping['pixel_x']: 'pixel_x',
            mapping['pixel_y']: 'pixel_y',
            mapping['recording_session']: 'recording_session',
            mapping['page_name']: 'page_name',
        }

        if mapping['time']:
            column_mapping[mapping['time']] = 'time'

        if mapping['duration']:
            column_mapping[mapping['duration']] = 'duration'

        for filter_col in mapping['filter_columns']:
            column_mapping[filter_col] = filter_col

        try:
            csv_files = [
                f for f in Path(dataset_path).glob(
                    '*.csv',
                ) if 'fixfinal' in f.name
            ]

            if not csv_files:
                print(f"ERROR: No valid CSV file found in {dataset_path}!")
                return

            csv_file = csv_files[0]
            print(f"Loading gaze data from: {csv_file}")

            base_df = pd.read_csv(
                csv_file,
                sep=None,
                engine='python',
                encoding='utf-8-sig',
                usecols=list(column_mapping.keys()),
            )
            base_df.rename(columns=column_mapping, inplace=True)
            base_df['normalized_page_name'] = base_df['page_name'].astype(
                str).apply(self._normalize_stimulus_name)

            for session in recording_sessions:
                filter_conditions = (
                    (base_df['recording_session'] == session) &
                    (base_df['normalized_page_name']
                     == normalized_stimulus_name)
                )

                for col, allowed in mapping['filter_columns'].items():
                    if col not in base_df.columns:
                        print(
                            f"WARNING: Filter column '{col}' not found in data; ignoring filter.",
                        )
                        continue
                    filter_conditions &= base_df[col].isin(allowed)

                session_df = base_df[filter_conditions].copy()

                if session_df.empty:
                    print(
                        f"WARNING: No data for session '{session}' and stimulus '{normalized_stimulus_name}' found",
                    )
                    continue

                if 'time' in session_df.columns:
                    session_df.sort_values(by='time', inplace=True)
                elif 'duration' in session_df.columns:
                    session_df['time'] = session_df['duration'].cumsum().shift(
                        fill_value=0,
                    )
                    session_df.sort_values(by='time', inplace=True)
                else:
                    print(
                        "ERROR: Neither 'time' nor 'duration' column found after mapping.",
                    )
                    continue

                session_df['pixel'] = list(
                    zip(session_df['pixel_x'], session_df['pixel_y']),
                )

                self._normalize_timestamps(session_df)
                self.gaze_dfs.append((session, session_df))

        except (pd.errors.ParserError, FileNotFoundError) as e:
            print(f"ERROR: Failed to load gaze data - {e}")

        if self.is_image:
            self.image = cv2.imread(self.stimulus_path)

    def _normalize_stimulus_name(self, name: str) -> str:
        """Return the base name **without** extension."""
        return Path(name).stem.lower()

    def _is_image(self):
        """Check if the provided stimulus is an image."""
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        return self.stimulus_path.lower().endswith(image_extensions)

    def _extract_pixel_coordinates(self, pixel_value):
        """Extract and scale pixel coordinates to fit the stimulus resolution."""
        # Determine stimulus size
        if self.is_image:
            if self.image is None:
                print('ERROR: Image stimulus is missing!')
                return None
            stimulus_height, stimulus_width = self.image.shape[:2]
        else:
            capture = cv2.VideoCapture(self.stimulus_path)
            stimulus_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            stimulus_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            capture.release()

        # Validate and extract gaze coordinates
        if isinstance(pixel_value, (list, tuple, np.ndarray)) and len(pixel_value) == 2:
            try:
                # Keep as float before scaling
                x, y = float(pixel_value[0]), float(pixel_value[1])

                # Ensure coordinates fit within the stimulus resolution
                x = int(np.clip(x, 0, stimulus_width - 1))
                y = int(np.clip(y, 0, stimulus_height - 1))
                return x, y

            except (ValueError, TypeError) as e:
                print(
                    f"ERROR converting pixel data: {e}, Value: {pixel_value}",
                )
                return None

        print(f"Invalid gaze data: {pixel_value} (Type: {type(pixel_value)})")
        return None

    def _normalize_timestamps(self, df: pd.DataFrame):
        """Convert timestamps into corresponding frame indices."""
        if self.is_image:
            # All gaze points belong to a single frame
            df['frame_idx'] = 0
            return

        self.stimulus_path = os.path.abspath(self.stimulus_path)
        print(f"Checking video path: {self.stimulus_path}")

        capture = cv2.VideoCapture(self.stimulus_path)
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        print(
            f"Video loaded successfully! FPS: {fps}, Total Frames: {frame_count}",
        )

        required_columns = ['time']
        if not all(col in df.columns for col in required_columns):
            print(
                f"ERROR: Required columns {required_columns} not found in gaze_df!",
            )
            return

        # Normalize timestamps: shift them to start from 0
        min_time = df['time'].min()  # Get the first timestamp
        df['normalized_time'] = (df['time'] - min_time) / \
            1000.0  # Convert ms → s

        # Convert timestamps to frame indices using FPS
        df['frame_idx'] = np.clip(
            (df['normalized_time'] * fps).astype(int), 0, frame_count - 1,
        )

    def play(self, speed: float = 1.0) -> None:
        """Play the stimulus (video or image) with gaze overlay.

        Parameters
        ----------
        speed : float, optional
            Playback speed multiplier. Default is 1.0.
            Use values < 1.0 to slow down or > 1.0 to speed up playback.
        """
        if not self.gaze_dfs:
            print('ERROR: No gaze data loaded!')
            return

        if self.is_image:
            self._play_image_stimulus(speed)
        else:
            self._play_video_stimulus(speed)

    def _play_image_stimulus(self, speed: float):
        """Handle gaze playback for an image stimulus."""
        if self.image is None:
            print('ERROR: Failed to load image stimulus.')
            return

        speed_adjusted_fps = 30 * speed
        for frame in self._overlay_gaze_on_image(speed_adjusted_fps):
            cv2.imshow('Eye-Tracking Replay', frame)
            if cv2.waitKey(int(1000 / speed_adjusted_fps)) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def _play_video_stimulus(self, speed: float):
        """Handle gaze playback for a video stimulus."""
        capture = cv2.VideoCapture(self.stimulus_path)

        frame_idx = 0
        speed_adjusted_fps = 30 * speed
        while True:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = capture.read()
            if not ret:
                break  # Stop playback if no more frames are available

            current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            self._overlay_gaze_on_video(frame, current_frame)

            cv2.imshow('Eye-Tracking Replay', frame)
            if cv2.waitKey(int(1000 / speed_adjusted_fps)) & 0xFF == ord('q'):
                break

            frame_idx += 1

        capture.release()
        cv2.destroyAllWindows()

    def fixation_navigation(self):
        """Navigate through fixations with manual controls.

        Displays each fixation one at a time. Press:
        - `n` for next fixation
        - `p` for previous fixation
        - `q` to quit navigation
        """
        df = self._select_recording_session()
        if df is None:
            return

        if self.is_image:
            self._navigate_fixations_on_image(df)
        else:
            self._navigate_fixations_on_video(df)

    def _select_recording_session(self) -> pd.DataFrame | None:
        """Open a Tk dialog to pick a loaded recording session."""
        if not self.gaze_dfs:
            print('No recording sessions loaded.')
            return None

        root = tk.Tk()
        root.withdraw()
        sessions = [name for name, _ in self.gaze_dfs]
        choice = SessionSelectDialog(root, sessions).result
        root.destroy()

        if choice is None:
            return None

        for name, df in self.gaze_dfs:
            if name == choice:
                return df

        return None

    def _navigate_fixations_on_image(self, df: pd.DataFrame):
        """Handle fixation navigation for an image stimulus."""
        if self.image is None:
            print('ERROR: Failed to load image stimulus.')
            return

        fixations = df[df['pixel'].notna()]
        if fixations.empty:
            print('ERROR: No valid fixations found!')
            return

        idx = 0
        while True:
            frame = self.image.copy()  # Keep original image untouched

            pixel_coords = self._extract_pixel_coordinates(
                fixations.iloc[idx]['pixel'],
            )
            if pixel_coords is None:
                idx = min(idx + 1, len(fixations) - 1)
                continue

            cv2.circle(
                frame, pixel_coords, self.dot_radius,
                self.overlay_colors[0], -1,
            )

            self._draw_progress(frame, idx + 1, len(fixations))

            cv2.imshow('Fixation Navigation', frame)
            key = cv2.waitKey(10)

            if key == ord('n') and idx < len(fixations) - 1:
                idx += 1  # Next fixation
            elif key == ord('p') and idx > 0:
                idx -= 1  # Previous fixation
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

    def _navigate_fixations_on_video(self, df: pd.DataFrame):
        """Handle fixation navigation for a video stimulus."""
        capture = cv2.VideoCapture(self.stimulus_path)

        fixations = df[df['pixel'].notna()]
        # remove *consecutive* duplicates of the same frame_idx
        dedup_mask = fixations['frame_idx'].diff().fillna(1).ne(0)
        fixations = fixations[dedup_mask].reset_index(drop=True)
        if fixations.empty:
            print('ERROR: No valid fixations found!')
            return

        idx = 0
        while True:
            if idx >= len(fixations):
                print('Warning: No more fixations available!')
                break

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                fixations.iloc[idx]['frame_idx'],
            )
            ret, frame = capture.read()
            if not ret:
                break

            pixel_coords = self._extract_pixel_coordinates(
                fixations.iloc[idx]['pixel'],
            )
            if pixel_coords is None:
                idx += 1
                continue

            cv2.circle(
                frame, pixel_coords, self.dot_radius,
                self.overlay_colors[0], -1,
            )

            self._draw_progress(frame, idx + 1, len(fixations))

            cv2.imshow('Fixation Navigation', frame)
            key = cv2.waitKey(0)  # Wait for key press

            if key == ord('n') and idx < len(fixations) - 1:
                idx += 1  # Next fixation
            elif key == ord('p') and idx > 0:
                idx -= 1  # Previous fixation
            elif key == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

    def export_replay(self, filename: str, speed: float = 1.0):
        """Export the gaze replay as an MP4 video.

        Parameters
        ----------
        filename : str
            Name of the output file without extension. '.mp4' will be added automatically.
        speed : float, optional
            Playback speed multiplier. Default is 1.0.
            Use values < 1.0 to slow down or > 1.0 to speed up playback.
        """
        output_path = f"{filename}.mp4"
        speed_adjusted_fps = 30 * speed
        if self.is_image:
            self._export_replay_image_stimulus(output_path, speed_adjusted_fps)
        else:
            self._export_replay_video_stimulus(output_path, speed_adjusted_fps)

    def _export_replay_image_stimulus(self, output_path: str, fps: float):
        """Handle exporting gaze replay for an image stimulus as an MP4 video."""
        print('Exporting gaze replay for an image stimulus...')

        height, width = self.image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in self._overlay_gaze_on_image(fps):
            out.write(frame)

        out.release()
        print(f"Image-based replay exported as MP4: {output_path}")

    def _export_replay_video_stimulus(self, output_path: str, fps: float):
        """Handle exporting gaze replay for a video stimulus as an MP4 video."""
        print('Exporting gaze replay for a video stimulus...')

        capture = cv2.VideoCapture(self.stimulus_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = capture.read()
            if not ret:
                break  # Stop playback if no more frames are available

            current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            self._overlay_gaze_on_video(frame, current_frame)
            out.write(frame)

        capture.release()
        out.release()
        print(f"Video-based replay exported as MP4: {output_path}")

    def _overlay_gaze_on_image(self, fps: float):
        # Tag each fixation with its session
        all_fixations = pd.concat(
            [df for _, df in self.gaze_dfs],
            keys=range(len(self.gaze_dfs)),
        )
        all_fixations.reset_index(inplace=True)
        all_fixations.sort_values(by='time', inplace=True)

        session_count = len(self.gaze_dfs)
        last_seen_fixation = [None] * session_count

        prev_time = 0.0
        for _, row in all_fixations.iterrows():
            session_index = row['level_0']
            current_time = row['time']
            delta_ms = max(current_time - prev_time, 1)
            prev_time = current_time

            last_seen_fixation[session_index] = row

            frame = self.image.copy()
            for i, fixation in enumerate(last_seen_fixation):
                if fixation is None:
                    continue

                pixel_coords = self._extract_pixel_coordinates(
                    fixation['pixel'],
                )
                if pixel_coords:
                    color = self.overlay_colors[i % len(self.overlay_colors)]
                    cv2.circle(frame, pixel_coords, self.dot_radius, color, -1)

            self._draw_legend(frame)
            num_frames = int((delta_ms / 1000.0) * fps)
            yield from [frame] * max(1, num_frames)

    def _overlay_gaze_on_video(self, frame, current_frame):
        i: int
        for i, (_, df) in enumerate(self.gaze_dfs):
            gaze_data = df[df['frame_idx'] == current_frame]
            if not gaze_data.empty:
                pixel_coords = self._extract_pixel_coordinates(
                    gaze_data.iloc[0]['pixel'],
                )
                if pixel_coords:
                    color = self.overlay_colors[i % len(self.overlay_colors)]
                    cv2.circle(frame, pixel_coords, self.dot_radius, color, -1)

            self._draw_legend(frame)

    def _draw_legend(self, frame):
        legend_height = 20 * len(self.gaze_dfs) + 10
        legend_width = 250
        legend = np.ones(
            (legend_height, legend_width, 3),
            dtype=np.uint8,
        ) * 255

        i: int
        for i, (session_name, _) in enumerate(self.gaze_dfs):
            color = self.overlay_colors[i % len(self.overlay_colors)]
            y = 10 + i * 20
            cv2.circle(legend, (10, y + 5), self.dot_radius, color, -1)
            cv2.putText(
                legend,
                session_name,
                (25, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        frame[10: 10 + legend.shape[0], 10: 10 + legend.shape[1]] = legend

    def _draw_progress(self, frame, current: int, total: int):
        text = f"{current} / {total}"

        (w, h), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )
        pad = 6
        badge = np.ones((h + baseline + 2 * pad, w + 2 *
                        pad, 3), dtype=np.uint8) * 255

        cv2.putText(
            badge,
            text,
            (pad, h + pad),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        frame[10: 10 + badge.shape[0], 10: 10 + badge.shape[1]] = badge

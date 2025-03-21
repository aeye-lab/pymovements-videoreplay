from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import os


class VideoPlayer:

    def __init__(self, stimulus_path: str, dataset_path: str, recording_session: str):
        """
        Initializes the VideoPlayer.

        Parameters:
        - stimulus_path (str): Path to the stimulus video.
        - dataset_path (str): Path to the eye-tracking dataset.
        - recording_session (str): Name of the recording session for filtering.
        """

        self.stimulus_path = stimulus_path
        stimulus_name = os.path.basename(stimulus_path)
        stimulus_name = os.path.splitext(stimulus_name)[0]  # Remove the extension
        self.is_image = self._is_image()
        self.gaze_df = None

        column_mapping = {
            "CURRENT_FIX_X": "pixel_x",  # Rename X-coordinate
            "CURRENT_FIX_Y": "pixel_y",  # Rename Y-coordinate
            "CURRENT_FIX_DURATION": "duration",  # This will be used to calculate time
            "page_name": "page_name",  # Used for filtering
            "RECORDING_SESSION_LABEL": "recording_session"  # Used for filtering
        }

        try:
            csv_files = [f for f in Path(dataset_path).glob("*.csv") if "fixfinal" in f.name]

            if not csv_files:
                print(f"ERROR: No valid CSV file found in {dataset_path}!")
                self.gaze_df = pd.DataFrame()  # Assign an empty DataFrame
                return

            # Select the first matching file
            csv_file = csv_files[0]
            print(f"Loading gaze data from: {csv_file}")

            # Read CSV into DataFrame
            self.gaze_df = pd.read_csv(csv_file, usecols=column_mapping.keys())

            # Rename columns
            self.gaze_df.rename(columns=column_mapping, inplace=True)

            # Filter based on stimulus name
            if "page_name" in self.gaze_df.columns:
                self.gaze_df = self.gaze_df[self.gaze_df["page_name"] == stimulus_name].copy()

            if "recording_session" in self.gaze_df.columns:
                self.gaze_df = self.gaze_df[self.gaze_df["recording_session"] == recording_session].copy()

            if self.gaze_df.empty:
                print(f"WARNING: No matching gaze data found for stimulus '{stimulus_name}'!")
                return

            # Compute Cumulative Time (Start at 0)
            self.gaze_df["time"] = self.gaze_df["duration"].cumsum().shift(fill_value=0)

            # Combine pixel columns
            self.gaze_df["pixel"] = list(zip(self.gaze_df["pixel_x"], self.gaze_df["pixel_y"]))

        except Exception as e:
            print(f"ERROR: Failed to load gaze data - {e}")
            self.gaze_df = pd.DataFrame()  # Assign an empty DataFrame in case of failure

        self._normalize_timestamps()

        if self.is_image:
            self.image = cv2.imread(self.stimulus_path)

    def _is_image(self):
        """Check if the provided stimulus is an image."""
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        return self.stimulus_path.lower().endswith(image_extensions)

    def _extract_pixel_coordinates(self, pixel_value):
        """
        Extracts (x, y) coordinates from the pixel column and scales them to fit
        the stimulus resolution (image or video).
        """

        # Determine stimulus size
        if self.is_image:
            if self.image is None:
                print("ERROR: Image stimulus is missing!")
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
                x, y = float(pixel_value[0]), float(pixel_value[1])  # Keep as float before scaling

                # Ensure coordinates fit within the stimulus resolution
                x = int(np.clip(x, 0, stimulus_width - 1))  # Clip values within width range
                y = int(np.clip(y, 0, stimulus_height - 1))  # Clip values within height range
                return x, y

            except (ValueError, TypeError) as e:
                print(f"ERROR converting pixel data: {e}, Value: {pixel_value}")
                return None

        print(f"Invalid gaze data: {pixel_value} (Type: {type(pixel_value)})")
        return None  # Return None for invalid values

    def _normalize_timestamps(self):
        """Converts timestamps into corresponding frame indices."""

        if self.is_image:
            self.gaze_df["frame_idx"] = 0  # All gaze points belong to a single frame
            return

        # Ensure correct video path
        self.stimulus_path = os.path.abspath(self.stimulus_path)
        print(f"Checking video path: {self.stimulus_path}")

        # Open video file
        capture = cv2.VideoCapture(self.stimulus_path)

        # Get FPS and total frame count
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        print(f"Video loaded successfully! FPS: {fps}, Total Frames: {frame_count}")

        # Check if required columns exist
        required_columns = ['time']
        if not all(col in self.gaze_df.columns for col in required_columns):
            print(f"ERROR: Required columns {required_columns} not found in gaze_df!")
            return

        # Normalize timestamps: shift them to start from 0
        min_time = self.gaze_df['time'].min()  # Get the first timestamp
        self.gaze_df['normalized_time'] = (self.gaze_df['time'] - min_time) / 1000.0  # Convert ms → s

        # Convert timestamps to frame indices using FPS
        self.gaze_df['frame_idx'] = np.clip((self.gaze_df['normalized_time'] * fps).astype(int), 0, frame_count - 1)

        print("frame_idx added!")
        print("Min Frame Index:", self.gaze_df["frame_idx"].min())
        print("Max Frame Index:", self.gaze_df["frame_idx"].max())

    def play(self, speed: float = 1.0):
        """
        Plays the stimulus (video or image) with gaze overlay.

        Parameters:
        - speed (float): Playback speed (1.0 = normal, <1.0 = slow, >1.0 = fast).
        """

        # Check if required columns exist
        required_columns = ['time', 'pixel', 'frame_idx']
        if not all(col in self.gaze_df.columns for col in required_columns):
            print(f"ERROR: Required columns {required_columns} not found in gaze_df!")
            return

        if self.is_image:
            self._play_image_stimulus(speed)
        else:
            self._play_video_stimulus(speed)

    def _play_image_stimulus(self, speed: float):
        """Handles gaze playback for an image stimulus."""
        if self.image is None:
            print("ERROR: Failed to load image stimulus.")
            return

        self.gaze_df.sort_values(by='time', inplace=True)
        for _, row in self.gaze_df.iterrows():
            frame = self.image.copy()  # Keep the original image untouched

            # Extract (x, y) coordinates
            pixel_coords = self._extract_pixel_coordinates(row['pixel'])
            if pixel_coords is None:
                continue  # Skip invalid gaze points

            x, y = pixel_coords
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Draw red dot
            cv2.imshow("Eye-Tracking Replay", frame)

            if cv2.waitKey(int(1000 / speed)) & 0xFF == ord('q'):  # Wait briefly per gaze point
                break

        cv2.destroyAllWindows()

    def _play_video_stimulus(self, speed: float):
        """Handles gaze playback for a video stimulus."""
        capture = cv2.VideoCapture(self.stimulus_path)

        frame_idx = 0
        while True:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, video_frame = capture.read()
            if not ret:
                break  # Stop playback if no more frames are available

            current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))

            # Get gaze data for the current frame
            gaze_data = self.gaze_df[self.gaze_df['frame_idx'] == current_frame]

            if not gaze_data.empty:
                pixel_coords = self._extract_pixel_coordinates(gaze_data.iloc[0]['pixel'])
                if pixel_coords is not None:
                    x, y = pixel_coords
                    cv2.circle(video_frame, (x, y), 5, (0, 0, 255), -1)  # Draw red dot

            # Show video frame with gaze overlay
            cv2.imshow('Eye-Tracking Replay', video_frame)

            # Exit playback on 'q' press
            if cv2.waitKey(int(1000 / (speed * 30))) & 0xFF == ord('q'):
                break

            frame_idx += 1  # Move to next frame

        capture.release()
        cv2.destroyAllWindows()

    def fixation_navigation(self):
        """Navigates through fixations in the eye-tracking data."""
        if self.is_image:
            self._navigate_fixations_on_image()
        else:
            self._navigate_fixations_on_video()

    def _navigate_fixations_on_image(self):
        """Handles fixation navigation for an image stimulus."""
        if self.image is None:
            print("ERROR: Failed to load image stimulus.")
            return

        fixations = self.gaze_df[self.gaze_df['pixel'].notna()]  # Filter valid fixations
        if fixations.empty:
            print("ERROR: No valid fixations found!")
            return

        idx = 0
        while True:
            frame = self.image.copy()  # Keep original image untouched

            pixel_coords = self._extract_pixel_coordinates(fixations.iloc[idx]['pixel'])
            if pixel_coords is None:
                idx = min(idx + 1, len(fixations) - 1)
                continue

            x, y = pixel_coords
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)  # Green dot for fixations
            cv2.imshow("Fixation Navigation", frame)
            key = cv2.waitKey(10)

            if key == ord('n') and idx < len(fixations) - 1:
                idx += 1  # Next fixation
            elif key == ord('p') and idx > 0:
                idx -= 1  # Previous fixation
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

    def _navigate_fixations_on_video(self):
        """Handles fixation navigation for a video stimulus."""
        capture = cv2.VideoCapture(self.stimulus_path)

        fixations = self.gaze_df[self.gaze_df['pixel'].notna()]  # Filter valid fixations
        if fixations.empty:
            print("ERROR: No valid fixations found!")
            return

        idx = 0
        while True:
            if idx >= len(fixations):
                print("Warning: No more fixations available!")
                break

            capture.set(cv2.CAP_PROP_POS_FRAMES, fixations.iloc[idx]['frame_idx'])
            frame_read_successful, video_frame = capture.read()
            if not frame_read_successful:
                break

            pixel_coords = self._extract_pixel_coordinates(fixations.iloc[idx]['pixel'])
            if pixel_coords is None:
                idx += 1  # Skip invalid fixation
                continue

            x, y = pixel_coords
            cv2.circle(video_frame, (x, y), 8, (0, 255, 0), -1)  # Green dot for fixations (BGR format)

            cv2.imshow('Fixation Navigation', video_frame)
            key = cv2.waitKey(0)  # Wait for key press

            if key == ord('n') and idx < len(fixations) - 1:
                idx += 1  # Next fixation
            elif key == ord('p') and idx > 0:
                idx -= 1  # Previous fixation
            elif key == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

    def export_replay(self, filename: str, fps: int = 30):
        """
        Exports the gaze replay as an MP4 file (for both videos and images).

        The user only needs to provide the filename (without an extension).
        The correct extension is automatically determined.

        Parameters:
        - filename (str): Name of the output file (without extension).
        - fps (int): Frames per second for the exported video.
        """

        output_path = f"{filename}.mp4"
        if self.is_image:
            self._export_replay_image_stimulus(output_path, fps)
        else:
            self._export_replay_video_stimulus(output_path, fps)

    def _export_replay_image_stimulus(self, output_path: str, fps: int):
        """Exports gaze replay from image stimulus as an MP4 video."""
        print("Exporting gaze replay for an image stimulus...")

        height, width = self.image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.gaze_df.sort_values(by='time', inplace=True)

        for _, row in self.gaze_df.iterrows():
            frame = self.image.copy()
            pixel_coords = self._extract_pixel_coordinates(row['pixel'])

            if pixel_coords:
                x, y = pixel_coords
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            out.write(frame)

        out.release()
        print(f"Image-based replay exported as MP4: {output_path}")

    def _export_replay_video_stimulus(self, output_path: str, fps: int):
        """Handles exporting gaze replay for a video stimulus as an MP4."""
        print("Exporting gaze replay for a video stimulus...")

        capture = cv2.VideoCapture(self.stimulus_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, video_frame = capture.read()
            if not ret:
                break  # Stop playback if no more frames are available

            current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))

            # Get gaze data for the current frame
            gaze_data = self.gaze_df[self.gaze_df['frame_idx'] == current_frame]

            if not gaze_data.empty:
                pixel_coords = self._extract_pixel_coordinates(gaze_data.iloc[0]['pixel'])

                if pixel_coords:
                    x, y = pixel_coords
                    cv2.circle(video_frame, (x, y), 5, (0, 0, 255), -1)  # Draw red dot

            out.write(video_frame)  # Write frame to video

        capture.release()
        out.release()
        print(f"Video-based replay exported as MP4: {output_path}")

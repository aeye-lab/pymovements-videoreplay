import cv2
import imageio
import numpy as np
import pymovements as pm
import os


class VideoPlayer:

    def __init__(self, video_path: str, dataset_path: str, dataset_name: str):
        """
        Initializes the VideoPlayer.

        Parameters:
        - video_path (str): Path to the stimulus video.
        - dataset_path (str): Path to the eye-tracking dataset.
        - dataset_name (str): Name of the dataset definition in pymovements.
        """

        self.video_path = video_path
        self.dataset = pm.Dataset(definition=dataset_name, path=dataset_path)
        self.dataset.load()
        self.gaze_df = self.dataset.gaze[0].frame.to_pandas()
        self._normalize_timestamps()

    def _normalize_timestamps(self):
        """Converts timestamps into corresponding frame indices."""

        # Ensure correct video path
        self.video_path = os.path.abspath(self.video_path)
        print(f"Checking video path: {self.video_path}")

        # Open video file
        capture = cv2.VideoCapture(self.video_path)

        # Debug: Check if the video opens correctly
        if not capture.isOpened():
            print(f"ERROR: Failed to open video file: {self.video_path}")
            return  # Exit the function

        # Get FPS and total frame count
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        # Debug: Ensure valid FPS and frame count
        if abs(fps) < 1e-6 or frame_count == 0:  # Floating-point numbers (fps) should not be compared using == directly
            print(f"ERROR: FPS is {fps}, Total Frames: {frame_count}")
            return  # Exit the function

        print(f"Video loaded successfully! FPS: {fps}, Total Frames: {frame_count}")

        # Check if required columns exist
        print(f"Available columns in gaze_df: {self.gaze_df.columns.tolist()}")
        required_columns = ['time']
        if not all(col in self.gaze_df.columns for col in required_columns):
            print(f"ERROR: Required columns {required_columns} not found in gaze_df!")
            return

        # Normalize timestamps: shift them to start from 0
        min_time = self.gaze_df['time'].min()  # Get the first timestamp
        self.gaze_df['normalized_time'] = (self.gaze_df['time'] - min_time) / 1000.0  # Convert ms → s

        # Convert timestamps to frame indices using FPS
        self.gaze_df['frame_idx'] = np.clip((self.gaze_df['normalized_time'] * fps).astype(int), 0, frame_count - 1)

        print("frame_idx added! (Now properly scaled)")
        print(self.gaze_df[['time', 'normalized_time', 'frame_idx']].head())  # Debugging output
        print("🕒 Min Frame Index:", self.gaze_df["frame_idx"].min())
        print("🕒 Max Frame Index:", self.gaze_df["frame_idx"].max())

    def play(self, speed: float = 1.0):
        """
        Plays the stimulus video with gaze overlay.

        Parameters:
        - speed (float): Playback speed (1.0 = normal, <1.0 = slow, >1.0 = fast).
        """

        # Check if required columns exist
        print(f"Available columns in gaze_df: {self.gaze_df.columns.tolist()}")
        required_columns = ['frame_idx', 'pixel']
        if not all(col in self.gaze_df.columns for col in required_columns):
            print(f"ERROR: Required columns {required_columns} not found in gaze_df!")
            return

        capture = cv2.VideoCapture(self.video_path)

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
                pixel_value = gaze_data.iloc[0]['pixel']  # Extract pixel value

                # Handle different formats: List, NumPy array, Tuple
                if isinstance(pixel_value, (list, tuple, np.ndarray)) and len(pixel_value) == 2:
                    pixel_x, pixel_y = int(pixel_value[0]), int(pixel_value[1])

                    # Debugging output:
                    print(f"Frame {current_frame} | Gaze: X={pixel_x}, Y={pixel_y}")

                    # Draw the red dot at the gaze position
                    cv2.circle(video_frame, (pixel_x, pixel_y), 5, (0, 0, 255), -1)
                else:
                    print(f"Invalid gaze data at Frame {current_frame}: {pixel_value}")

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
        capture = cv2.VideoCapture(self.video_path)
        fixations = self.gaze_df[self.gaze_df['stimuli_x'] != -1]  # Filter valid fixations
        idx = 0

        while True:
            capture.set(cv2.CAP_PROP_POS_FRAMES, fixations.iloc[idx]['frame_idx'])
            frame_read_successful, video_frame = capture.read()
            if not frame_read_successful:
                break

            x, y = int(fixations.iloc[idx]['x']), int(fixations.iloc[idx]['y'])
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

    def export_replay(self, output_path: str):
        """
        Exports the gaze replay as an MP4 file.

        Parameters:
        - output_path (str): Path where the replay video should be saved.
        """
        capture = cv2.VideoCapture(self.video_path)
        frames = []

        for _, row in self.gaze_df.iterrows():
            capture.set(cv2.CAP_PROP_POS_FRAMES, row['frame_idx'])
            frame_read_successful, video_frame = capture.read()
            if not frame_read_successful:
                break

            x, y = int(row['x']), int(row['y'])
            cv2.circle(video_frame, (x, y), 5, (0, 0, 255), -1)  # Red dot for gaze (BGR format)

            frames.append(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))  # Convert for imageio

        capture.release()
        imageio.mimsave(output_path, frames, fps=25)
        print(f"Replay saved to {output_path}")

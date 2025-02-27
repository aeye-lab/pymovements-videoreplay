import cv2
import imageio
import numpy as np
import pymovements as pm
import os


def _is_image(file_path):
    """Check if the provided stimulus is an image."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return file_path.lower().endswith(image_extensions)


def _extract_pixel_coordinates(pixel_value):
    """Extracts (x, y) coordinates from the pixel column, ensuring proper format."""
    if isinstance(pixel_value, (list, tuple, np.ndarray)) and len(pixel_value) == 2:
        return int(pixel_value[0]), int(pixel_value[1])
    print(f"Invalid gaze data: {pixel_value}")
    return None  # Return None for invalid values


class VideoPlayer:

    def __init__(self, stimulus_path: str, dataset_path: str, dataset_name: str):
        """
        Initializes the VideoPlayer.

        Parameters:
        - video_path (str): Path to the stimulus video.
        - dataset_path (str): Path to the eye-tracking dataset.
        - dataset_name (str): Name of the dataset definition in pymovements.
        """

        self.stimulus_path = stimulus_path
        self.is_image = _is_image(stimulus_path)

        self.dataset = pm.Dataset(definition=dataset_name, path=dataset_path)
        self.dataset.load()
        self.gaze_df = self.dataset.gaze[0].frame.to_pandas()
        self._normalize_timestamps()

        if self.is_image:
            self.image = cv2.imread(self.stimulus_path)

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

        # Debug: Check if the video opens correctly
        if not capture.isOpened():
            print(f"ERROR: Failed to open video file: {self.stimulus_path}")
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
        Plays the stimulus (video or image) with gaze overlay.

        Parameters:
        - speed (float): Playback speed (1.0 = normal, <1.0 = slow, >1.0 = fast).
        """

        # Check if required columns exist
        print(f"Available columns in gaze_df: {self.gaze_df.columns.tolist()}")
        required_columns = ['frame_idx', 'pixel']
        if not all(col in self.gaze_df.columns for col in required_columns):
            print(f"ERROR: Required columns {required_columns} not found in gaze_df!")
            return

        if self.is_image:
            self._play_image_stimulus()
        else:
            self._play_video_stimulus(speed)

    def _play_image_stimulus(self):
        """Handles gaze playback for an image stimulus."""
        if self.image is None:
            print("ERROR: Failed to load image stimulus.")
            return

        for _, row in self.gaze_df.iterrows():
            frame = self.image.copy()  # Keep the original image untouched

            # Extract (x, y) coordinates
            pixel_coords = _extract_pixel_coordinates(row['pixel'])
            if pixel_coords is None:
                continue  # Skip invalid gaze points

            x, y = pixel_coords
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Draw red dot
            cv2.imshow("Eye-Tracking Replay", frame)

            if cv2.waitKey(200) & 0xFF == ord('q'):  # Wait briefly per gaze point
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
                pixel_coords = _extract_pixel_coordinates(gaze_data.iloc[0]['pixel'])
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

            # Extract (x, y) coordinates
            pixel_coords = _extract_pixel_coordinates(fixations.iloc[idx]['pixel'])
            if pixel_coords is None:
                idx += 1  # Skip invalid fixation
                continue

            x, y = pixel_coords
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)  # Green dot for fixations

            cv2.imshow("Fixation Navigation", frame)
            key = cv2.waitKey(0)

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

            # Extract (x, y) coordinates
            pixel_coords = _extract_pixel_coordinates(fixations.iloc[idx]['pixel'])
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

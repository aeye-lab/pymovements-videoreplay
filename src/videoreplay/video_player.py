import cv2
import imageio
import pymovements as pm


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
        self.gaze_df = self.dataset.gaze[0].to_pandas()
        self._normalize_timestamps()

    def _normalize_timestamps(self):
        """Converts timestamps into corresponding frame indices."""
        capture = cv2.VideoCapture(self.video_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        min_time, max_time = self.gaze_df['timestamp'].min(), self.gaze_df['timestamp'].max()
        self.gaze_df['frame_idx'] = ((self.gaze_df['timestamp'] - min_time) /
                                     (max_time - min_time) * frame_count).astype(int)
        capture.release()

    def play(self, speed: float = 1.0):
        """
        Plays the stimulus video with gaze overlay.

        Parameters:
        - speed (float): Playback speed (1.0 = normal, <1.0 = slow, >1.0 = fast).
        """
        capture = cv2.VideoCapture(self.video_path)
        frame_idx = 0

        while capture.isOpened():
            frame_read_successful, video_frame = capture.read()
            if not frame_read_successful:
                break

            if frame_idx in self.gaze_df['frame_idx'].values:
                gaze_point = self.gaze_df[self.gaze_df['frame_idx'] == frame_idx].iloc[0]
                x, y = int(gaze_point['x']), int(gaze_point['y'])
                cv2.circle(video_frame, (x, y), 5, (0, 0, 255), -1)  # Red dot for gaze (BGR format)

            cv2.imshow('Eye-Tracking Replay', video_frame)
            if cv2.waitKey(int(25 / speed)) & 0xFF == ord('q'):
                break

            frame_idx += 1

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

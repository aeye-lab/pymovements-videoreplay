import cv2
import matplotlib.pyplot as plt


class VideoPlayer:
    def __init__(self, stimulus_path, gaze_data):
        self.stimulus_path = stimulus_path
        self.gaze_data = gaze_data  # Processed gaze data (e.g., fixations/saccades)

    def play(self, speed=1.0, event_based=True):
        # Logic to play the video with gaze overlay
        pass

    def export(self, output_path):
        # Logic to export the video with overlays
        pass

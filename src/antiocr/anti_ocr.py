from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


class AntiOCR:
    """Generates a stimulus image using words placed at given pixel coordinates.

    Parameters
    ----------
    frame_width : int
        Width of the output image.
    frame_height : int
        Height of the output image.
    font_scale : float, optional
        Font scale for the text rendering (default is 1.0).
    font_color : tuple[int, int, int], optional
        Font color in BGR format (default is black).
    font_thickness : int, optional
        Thickness of the font (default is 1).
    """

    def __init__(
            self,
            frame_width: int,
            frame_height: int,
            font_scale: float = 1.0,
            font_color: tuple[int, int, int] = (0, 0, 0),
            font_thickness: int = 1,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.font_scale = font_scale
        self.font_color = font_color
        self.font_thickness = font_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def generate_from_csv(self, csv_path: str, page_name: str, session: str, output_path: str):
        """Generate a stimulus image using word annotations from a CSV file.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file with columns: 'pixel_x', 'pixel_y', 'word'.
        output_path : str
            File path to save the generated image (should end in .png or .jpg).
        """
        try:
            csv_files = [f for f in Path(csv_path).glob(
                '*.csv') if 'fixfinal' in f.name]
            if not csv_files:
                print(f"ERROR: No valid CSV file found in {csv_path}!")
                return

            csv_file = csv_files[0]
            print(f"Loading gaze data from: {csv_file}")

            column_mapping = {
                'CURRENT_FIX_X': 'pixel_x',  # Rename X-coordinate
                'CURRENT_FIX_Y': 'pixel_y',  # Rename Y-coordinate
                # Label which should be displayed at that coordinate
                'CURRENT_FIX_INTEREST_AREA_LABEL': 'word',
                'page_name': 'page_name',  # Used for filtering
                'RECORDING_SESSION_LABEL': 'recording_session',  # Used for filtering
            }
            df = pd.read_csv(csv_file, usecols=list(column_mapping.keys()))
            df.rename(columns=column_mapping, inplace=True)
            df = df[(df['page_name'] == page_name) & (
                df['recording_session'] == session)]

        except Exception as e:
            print(f"Failed to load CSV: {e}")
            return

        print(df.columns)
        if not {'pixel_x', 'pixel_y', 'word'}.issubset(df.columns):
            print("CSV must contain 'pixel_x', 'pixel_y', and 'word' columns.")
            return

        image = np.ones((self.frame_height, self.frame_width, 3),
                        dtype=np.uint8) * 255
        for _, row in df.iterrows():
            x, y = int(row['pixel_x']), int(row['pixel_y'])
            word = str(row['word'])

            cv2.putText(
                image,
                word,
                (x, y),
                self.font,
                self.font_scale,
                self.font_color,
                self.font_thickness,
                lineType=cv2.LINE_AA,
            )

        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        if not any(str(output_path).lower().endswith(ext) for ext in valid_extensions):
            print(
                'ERROR: Output file must have a valid image extension (e.g., .png or .jpg)')
            return

        success = cv2.imwrite(output_path, image)
        if success:
            print(f"Stimulus image successfully saved to: {output_path}")
        else:
            print(f"Failed to save image to: {output_path}")

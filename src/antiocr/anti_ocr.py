"""Rebuild stimulus images from eye-tracking CSV data (reverse OCR).

`AntiOCR` turns the usual OCR pipeline on its head: given each word and its
recorded pixel coordinates, it recreates the original page as a flat image.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from antiocr.column_mapping_dialog import ColumnMappingDialog
# ── standard library ────────────────────────────────────────────────
# ── third-party libraries ───────────────────────────────────────────
# ── local package ───────────────────────────────────────────────────


class AntiOCR:
    """
    Render a stimulus image by drawing each `word` from an eye-tracking CSV
    at its recorded pixel coordinates.

    Parameters
    ----------
    frame_width, frame_height : int
        Output image dimensions in pixels.
    font_scale : float, default 1.0
        OpenCV font scale for the text.
    font_color : tuple[int, int, int], default (0, 0, 0)
        Text colour in BGR.
    font_thickness : int, default 1
        Stroke thickness used by ``cv2.putText``.
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
        """Create and save a stimulus image from a word-annotation CSV.

        Parameters
        ----------
        csv_path : str
            Directory containing the eye-tracking CSV export;
        page_name : str
            Page/stimulus identifier to extract.
        session : str
            Recording-session label to extract.
        output_path : str
            Destination image file; must end with ``.png``, ``.jpg``, ``.jpeg`` or
            ``.bmp``.
        """
        root = tk.Tk()
        root.withdraw()
        mapping = ColumnMappingDialog(root, title='Column Mapping').result
        root.destroy()

        if mapping is None:
            raise ValueError('Column mapping configuration cancelled by user.')

        column_mapping = {
            mapping['pixel_x']: 'pixel_x',
            mapping['pixel_y']: 'pixel_y',
            mapping['interest_area_label']: 'word',
            mapping['recording_session']: 'recording_session',
            mapping['page_name']: 'page_name',
        }

        for filter_col in mapping['filter_columns']:
            column_mapping[filter_col] = filter_col

        try:
            csv_files = [
                f for f in Path(csv_path).glob(
                    '*.csv',
                ) if 'fixfinal' in f.name
            ]

            if not csv_files:
                print(f"ERROR: No valid CSV file found in {csv_path}!")
                return

            csv_file = csv_files[0]
            print(f"Loading gaze data from: {csv_file}")

            df = pd.read_csv(
                csv_file,
                sep=None,
                engine='python',
                encoding='utf-8-sig',
                usecols=list(column_mapping.keys())
            )
            df.rename(columns=column_mapping, inplace=True)

            filter_conditions = (
                (df['recording_session'] == session) &
                (df['page_name'] == page_name)
            )

            for col, allowed in mapping['filter_columns'].items():
                if col not in df.columns:
                    print(
                        f"WARNING: Filter column '{col}' not found in data; ignoring filter.",
                    )
                    continue
                filter_conditions &= df[col].isin(allowed)

            df = df[filter_conditions].copy()

        except Exception as e:
            print(f"Failed to load CSV: {e}")
            return

        image = np.ones(
            (self.frame_height, self.frame_width, 3),
            dtype=np.uint8,
        ) * 255

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
                'ERROR: Output file must have a valid image extension (e.g., .png or .jpg)',
            )
            return

        success = cv2.imwrite(output_path, image)
        if success:
            print(f"Stimulus image successfully saved to: {output_path}")
        else:
            print(f"Failed to save image to: {output_path}")

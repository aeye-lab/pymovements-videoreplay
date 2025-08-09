# Copyright (c) 2025 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Rebuild stimulus images from CSV data via reverse OCR.

`AntiOCR` turns the usual OCR pipeline on its head: given each word and its
recorded pixel coordinates, it recreates the original page as a flat image.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from antiocr.column_mapping_dialog import ColumnMappingDialog


class AntiOCR:
    """Generate a stimulus image by plotting words from eye-tracking data.

    Reverse the OCR process by positioning each word at its original location.

    Parameters
    ----------
    frame_width: int
        Output image width in pixels.
    frame_height: int
        Output image height in pixels.
    font: int
        OpenCV font face used for rendering text.
        Must be one of OpenCV's font constants, e.g.,
        ``cv2.FONT_HERSHEY_SIMPLEX``, ``cv2.FONT_HERSHEY_PLAIN``, etc.
        (default: cv2.FONT_HERSHEY_SIMPLEX)
    font_scale: float
        OpenCV font scale for the text. (default: 1.0)
    font_color: tuple[int, int, int]
        Text colour in BGR. (default: (0, 0, 0))
    font_thickness: int
        Stroke thickness used by ``cv2.putText``. (default: 1)
    mapping: dict[str, str | dict[str, list[str]]] | None
        Optional column mapping for gaze data.
        If None, a column-mapping dialog will be shown to the user.
        The mapping must include keys for pixel_x, pixel_y,
        interest_area_label, recording_session, page_name,
        and optionally filter_columns.
        (default: None)
    """

    def __init__(
            self,
            frame_width: int,
            frame_height: int,
            font: int = cv2.FONT_HERSHEY_SIMPLEX,
            font_scale: float = 1.0,
            font_color: tuple[int, int, int] = (0, 0, 0),
            font_thickness: int = 1,
            mapping: dict[str, str | dict[str, list[str]]] | None = None,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.font = font
        self.font_scale = font_scale
        self.font_color = font_color
        self.font_thickness = font_thickness
        self.mapping = mapping

    def generate_from_csv(
            self,
            csv_path: str,
            page_name: str,
            session: str,
            output_path: str,
            custom_read_kwargs: dict[str, Any] | None = None,
    ) -> int:
        """Create and save a stimulus image from a word-annotation CSV.

        Parameters
        ----------
        csv_path: str
            Path to the eye-tracking CSV file.
        page_name: str
            Page/stimulus identifier to extract.
        session: str
            Recording-session label to extract.
        output_path: str
            Destination image file;
            must end with ``.png``, ``.jpg``, ``.jpeg`` or ``.bmp``.
        custom_read_kwargs: dict[str, Any] | None
            Additional keyword arguments passed to ``pandas.read_csv``.
            Useful for specifying delimiter (e.g., ``delimiter=','``),
            encoding, or handling different file formats.
            (default: None)

        Returns
        -------
        int
            0 if the image was successfully created and saved,
            1 if an error occurred during processing.

        Raises
        ------
        ValueError
            If the user cancels the column-mapping dialog.
        """
        mapping = self.mapping
        if mapping is None:
            root = tk.Tk()
            root.withdraw()
            mapping = ColumnMappingDialog(root, title='Column Mapping').result
            root.destroy()

            if mapping is None:
                raise ValueError(
                    'Column mapping configuration cancelled by user.',
                )

        column_mapping = {
            mapping['pixel_x']: 'pixel_x',
            mapping['pixel_y']: 'pixel_y',
            mapping['interest_area_label']: 'word',
            mapping['recording_session']: 'recording_session',
            mapping['page_name']: 'page_name',
        }

        for filter_col in mapping['filter_columns']:
            column_mapping[filter_col] = filter_col

        read_kwargs: dict[str, Any] = {
            'sep': None,
            'engine': 'python',
            'encoding': 'utf-8-sig',
            'usecols': list(column_mapping.keys()),
        }
        if custom_read_kwargs:
            read_kwargs.update(custom_read_kwargs)

        try:
            df = pd.read_csv(csv_path, **read_kwargs)
        except pd.errors.ParserError as e:
            print(f"ERROR: Failed to parse CSV - {e}")
            return 1
        except FileNotFoundError as e:
            print(f"ERROR: File not found - {e}")
            return 1

        df.rename(columns=column_mapping, inplace=True)
        df['normalized_page_name'] = df['page_name'].astype(str).apply(
            self._normalize_stimulus_name,
        )

        normalized_stimulus_name = self._normalize_stimulus_name(page_name)
        filter_conditions = (
            (df['recording_session'] == session) &
            (df['normalized_page_name'] == normalized_stimulus_name)
        )

        if isinstance(mapping['filter_columns'], dict):
            for col, allowed in mapping['filter_columns'].items():
                if col not in df.columns:
                    print(
                        f"WARNING: Filter column '{col}' not found; "
                        f"ignoring filter.",
                    )
                    continue
                filter_conditions &= df[col].isin(allowed)
        else:
            print(
                "WARNING: 'filter_columns' is not a dictionary; "
                'skipping filters.',
            )

        df = df[filter_conditions].copy()

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
        if not any(
                str(output_path).lower().endswith(ext)
                for ext in valid_extensions
        ):
            print(
                'ERROR: Output file must have a valid image extension '
                '(e.g., .png or .jpg)',
            )
            return 1

        success = cv2.imwrite(output_path, image)
        if success:
            print(f"Stimulus image successfully saved to: {output_path}")
            return 0

        print(f"Failed to save image to: {output_path}")
        return 1

    def _normalize_stimulus_name(self, name: str) -> str:
        """Return the base name **without** extension."""
        return Path(name).stem.lower()

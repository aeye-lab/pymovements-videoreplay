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
"""Correct recording artefacts in the fixations of eye-tracking data.

This module provides the `FixationCorrection` class
to manually correct fixations
and save the corrected data in a new file
"""
from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path

import cv2
import pandas as pd
import src.fixationcorrection.column_mapping_dialogue as cmd
import src.fixationcorrection.ocr_reader as ocr
from pynput import keyboard


class FixationCorrection:
    """A class to manually correct fixation points on an image.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    pandas_dataframe : pd.DataFrame
        A DataFrame containing fixation data.
    mapping : dict[str, str]
        Dictionary mapping image names or IDs to fixation data.
    title : str | None, optional
        Title for the visualization window.
    """

    def __init__(
        self, image_path: str, pandas_dataframe: pd.DataFrame,
        mapping: dict[str, str], title: str | None = None,
    ):
        self.image_path = image_path
        self.pandas_dataframe = pandas_dataframe
        self.mapping = mapping
        self.image = cv2.imread(self.image_path)
        self.current_fixation_index = 0  # Index of the current active circle
        self.fixation_coordinates = None  # Fixations for the current image
        self.title = title
        self.get_xy_coordinates()
        self.point_movement_mode = 1
        self.ocr_centers = None
        self.correction = True
        self.original_fixation = None

        self.get_ocr_centers()
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

    def get_xy_coordinates(self):
        """Extract (x, y) pixel coordinates from the DataFrame."""
        xy_coordinates = list(
            zip(
                self.pandas_dataframe[
                    self.mapping['pixel_x']
                ], self.pandas_dataframe[self.mapping['pixel_y']],
            ),
        )
        xy_int_coordinates = [(int(x), int(y)) for x, y in xy_coordinates]
        self.fixation_coordinates = xy_int_coordinates

    def draw_points_on_image(self):
        """Draw fixation points and connecting lines on the image.

        Highlight the current fixation in orange; others are purple
        """
        self.image = cv2.imread(self.image_path)

        for i, (x, y) in enumerate(self.fixation_coordinates):
            if (x, y) == (-1, -1):
                continue
            color = (0, 165, 255) if i == self.current_fixation_index else (
                (128, 0, 128)
            )
            cv2.circle(
                self.image, (x, y), radius=10, color=color,
                thickness=1,
            )  # Circle for points
            if i < len(self.fixation_coordinates):
                next_x, next_y = self.fixation_coordinates[
                    self.next_valid_fixation_index(
                        (i+1) % len(self.fixation_coordinates),
                    )
                ]
                cv2.line(
                    self.image, (next_x, next_y),
                    (x, y), color=color, thickness=1,
                )
        return self.image

    def move_point(self, direction):
        """Move the current fixation point.

        Either by one pixel (Pixel mode)
        or to the closest text bounding box (AOI mode)
        in the specified direction.
        """
        x, y = self.fixation_coordinates[self.current_fixation_index]

        if not self.original_fixation:
            self.original_fixation = (x, y)

        if self.point_movement_mode == 1:
            if direction == 'up':
                self.fixation_coordinates[self.current_fixation_index] = (
                    x, y - 1,
                )
            elif direction == 'down':
                self.fixation_coordinates[self.current_fixation_index] = (
                    x, y + 1,
                )  # Move down (increase y)
            elif direction == 'left':
                self.fixation_coordinates[self.current_fixation_index] = (
                    x - 1, y,
                )  # Move left (decrease x)
            elif direction == 'right':
                self.fixation_coordinates[self.current_fixation_index] = (
                    x + 1, y,
                )  # Move right (increase x)
        elif self.point_movement_mode == 0:
            if direction == 'up':
                self.fixation_coordinates[self.current_fixation_index] = (
                    ocr.find_closest_top_box(
                        x,
                        y,
                        self.ocr_centers,
                    )
                )
            elif direction == 'down':
                self.fixation_coordinates[self.current_fixation_index] = (
                    ocr.find_closest_bottom_box(
                        x,
                        y,
                        self.ocr_centers,
                    )
                )
            elif direction == 'left':
                self.fixation_coordinates[self.current_fixation_index] = (
                    ocr.find_closest_left_box(
                        x,
                        y,
                        self.ocr_centers,
                    )
                )
            elif direction == 'right':
                self.fixation_coordinates[self.current_fixation_index] = (
                    ocr.find_closest_right_box(
                        x,
                        y,
                        self.ocr_centers,
                    )
                )

    def delete_fixation(self):
        """Mark the current fixation as deleted.

        Set coordinates to (-1, -1).
        """
        x, y = self.fixation_coordinates[self.current_fixation_index]
        self.original_fixation = (x, y)

        self.fixation_coordinates[self.current_fixation_index] = (-1, -1)
        self.current_fixation_index = self.next_valid_fixation_index(
            self.current_fixation_index,
        )

    def undo_last_correction(self):
        """Restores the original position of the current fixation."""
        self.fixation_coordinates[self.current_fixation_index] \
            = self.original_fixation

    def on_press(self, key):
        """Handle key press events."""
        try:
            if key == keyboard.Key.up:
                self.move_point('up')
            elif key == keyboard.Key.down:
                self.move_point('down')
            elif key == keyboard.Key.left:
                self.current_fixation_index -= 1
                self.original_fixation = None
                self.current_fixation_index = (
                    self.previous_valid_fixation_index(
                        self.current_fixation_index,
                    )
                )
                if self.current_fixation_index < 0:
                    self.current_fixation_index = len(
                        self.fixation_coordinates,
                    ) - 1
            elif key == keyboard.Key.right:
                self.current_fixation_index += 1
                self.original_fixation = None
                self.current_fixation_index = self.next_valid_fixation_index(
                    self.current_fixation_index,
                )
                # Loop back to the first point
                if (
                    self.current_fixation_index >=
                    len(self.fixation_coordinates)
                ):
                    self.current_fixation_index = 0

            elif key.char == 'q':
                self.move_point('left')
            elif key.char == 'r':
                self.move_point('right')
            elif key.char == 'l':
                self.delete_fixation()
            elif key.char == 'z':
                self.undo_last_correction()
            elif key.char == 'm':
                self.switch_point_movement_mode()
            elif key.char == 'n':
                self.save_corrected_fixations()
                self.correction = False

        except AttributeError:
            pass

    def next_valid_fixation_index(self, index):
        """Find the next valid fixation of a given index."""
        n = len(self.fixation_coordinates)
        while self.is_invalid_fixation(self.fixation_coordinates[index]):
            index = (index + 1) % n

        return index

    def is_invalid_fixation(self, fixation):
        """Check if a fixation has been deleted (i.e., set to (-1, -1))."""
        if fixation == (-1, -1):
            return True
        return False

    def previous_valid_fixation_index(self, index):
        """Find the previous valid fixation of a given index."""
        n = len(self.fixation_coordinates)
        while self.is_invalid_fixation(self.fixation_coordinates[index]):
            index = (index - 1) % n
        return index

    def edit_points(self):
        """Launch the OpenCV image window.

        Enter a loop to allow user interaction for fixation correction.
        Close when user exits or completes correction.
        """
        while self.current_fixation_index < len(self.fixation_coordinates):
            # Draw the points on the image
            image_with_points = self.draw_points_on_image()
            self.display_point_movement_mode()
            # Display the image with the overlaid points
            cv2.imshow(f'Page {self.image_path}', image_with_points)
            cv2.setWindowTitle(
                f'Page {self.image_path}', self.image_path[
                    :-
                    4
                ] + ' ' + self.title,
            )

            cv2.waitKey(0)

            if not self.correction:
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()

    def save_corrected_fixations(self):
        """Save corrected fixation coordinates.

        Create new DataFrame columns, remove deleted fixations,
        and reset index.
        """
        self.pandas_dataframe[
            ['x_corrected', 'y_corrected']
        ] = pd.DataFrame(self.fixation_coordinates)
        self.pandas_dataframe = self.pandas_dataframe[
            ~(
                (
                    self.pandas_dataframe['x_corrected'] == -
                    1
                ) & (self.pandas_dataframe['y_corrected'] == -1)
            )
        ].copy()
        self.pandas_dataframe.reset_index(drop=True, inplace=True)

    def get_ocr_centers(self):
        """Initialize OCR reader and get center coordinates."""
        reader = ocr.OCR_Reader(self.image_path)
        reader.get_list_of_centers()
        self.ocr_centers = reader.list_of_centers

    def switch_point_movement_mode(self):
        """Toggle between Pixel- and AOI-based movement."""
        if self.point_movement_mode == 1:
            try:
                self.point_movement_mode = 0
            except ModuleNotFoundError:
                print("Feature unavailable: 'pytesseract' is not installed.")
        else:
            self.point_movement_mode = 1

    def display_point_movement_mode(self):
        """Overlay the current movement mode on the image."""
        mode = ''
        if self.point_movement_mode == 0:
            mode = 'AOI'
        elif self.point_movement_mode == 1:
            mode = 'Pixel'

        # Coordinates for box
        box_top_left = (50, 10)
        box_bottom_right = (360, 60)

        # Draw the box
        cv2.rectangle(
            self.image, box_top_left, box_bottom_right,
            (200, 200, 200), -1,
        )  # Grey background
        cv2.rectangle(
            self.image, box_top_left, box_bottom_right,
            (0, 0, 0), 1,
        )  # Black border

        # Add text
        cv2.putText(
            self.image, 'Fixation movement mode (m): ' +
            mode, (box_top_left[0] + 10, box_top_left[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )


class DataProcessing:
    """Handle loading, filtering, and grouping of CSV file.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing fixation data.
    image_folder : str
        Directory containing corresponding stimulus images.

    Raises
    ------
    ValueError
        If the user cancels the column-mapping dialog.
    """

    def __init__(self, csv_file: str, image_folder: str):
        self.csv_file = csv_file
        self.dataframes: list[pd.DataFrame] = []
        self.image_folder = image_folder
        self.image_list = os.listdir(self.image_folder)

        root = tk.Tk()
        root.withdraw()
        mapping = cmd.ColumnMappingDialog(root, title='Column Mapping').result
        root.destroy()

        if mapping is None:
            raise ValueError('Column mapping configuration cancelled by user.')

        self.column_mapping = {
            'pixel_x': mapping['pixel_x'],
            'pixel_y': mapping['pixel_y'],
            'image_column': mapping['image_column'],
            'grouping': mapping['grouping'],
            'filter_columns': mapping['filter_columns'],
        }

    def prepare_data(self):
        """Prepare data for fixation correction.

        Load the CSV file, remove rows without matching image files,
        and apply grouping and filtering as specified by the user.
        """
        raw_data = pd.read_csv(
            self.csv_file,
            sep=None,
            engine='python',
            encoding='utf-8-sig',
        )
        # Drop the entries where there is no corresponding image
        clean_list = {self.normalize(p) for p in self.image_list}
        mask = raw_data[self.column_mapping['image_column']].astype(
            str,
        ).apply(self.normalize).isin(clean_list)
        dropped = raw_data[mask]

        self.dataframes = self.filter_and_group(dropped)
        return self.dataframes

    def normalize(self, name: str) -> str:
        """Convert a filename to lowercase and strip its extension."""
        return Path(name).stem.lower()

    def filter_and_group(self, dataframe):
        """Filter and group the dataframe based on selected values."""
        self.make_title()
        if self.column_mapping['filter_columns'] is not None:
            for key, val in self.column_mapping['filter_columns'].items():
                if isinstance(val, list):
                    dataframe = dataframe[dataframe[key].isin(val)]
                else:
                    dataframe = dataframe[dataframe[key] == val]

        if self.column_mapping['grouping'] is not None:
            grouped = dataframe.groupby(self.column_mapping['grouping'])
            return [group.copy() for _, group in grouped]
        return [dataframe.copy()]

    def make_title(self):
        """Construct a title string from the selected filter values."""
        all_filters = []
        for value in self.column_mapping['filter_columns'].values():
            all_filters.append(value)
        flattened = [item for sublist in all_filters for item in sublist]
        title = '_'.join(flattened)
        return title


def run_fixation_correction(csv_file, image_folder):
    """Start the entire fixation correction process."""
    prepared = DataProcessing(
        csv_file, image_folder,
    )
    dataframes = prepared.prepare_data()
    mapping = prepared.column_mapping

    corrected_dataframes = []
    cv2.waitKey(0)

    for frame in dataframes:
        image_name = frame[mapping['image_column']].iloc[0]
        for image in os.listdir(image_folder):
            if image_name in image:
                image_path = os.path.join(image_folder, image)
                fix = FixationCorrection(
                    image_path, frame, mapping,
                    frame[mapping['grouping'][0]].iloc[0],
                )
                fix.edit_points()
                corrected_dataframes.append(fix.pandas_dataframe)

    # save the corrected data in a new file
    if corrected_dataframes:
        new_folder_name = 'corrected_fixations'
        combined_dataframe = pd.concat(corrected_dataframes, ignore_index=True)
        directory = os.path.dirname(csv_file)
        os.makedirs(os.path.join(directory, new_folder_name), exist_ok=True)
        filename = os.path.basename(csv_file)
        name, ext = os.path.splitext(filename)
        title = prepared.make_title()
        new_filename = f"{name}_fixation_corrected_{title}{ext}"
        new_path = os.path.join(directory, new_folder_name, new_filename)
        combined_dataframe.to_csv(new_path, index=False)




from __future__ import annotations

import os

import cv2
import ocr_reader
import pandas as pd


class FixationCorrection:
    def __init__(self, image_path, pandas_dataframe, title=None):
        self.image_path = image_path
        self.pandas_dataframe = pandas_dataframe
        self.image = cv2.imread(self.image_path)
        self.current_fixation_index = 0  # Index of the current active circle
        self.fixation_coordinates = None  # Fixations for the current image
        self.title = title
        self.last_deleted_fixation = None
        self.row_to_be_deleted = None
        self.get_xy_coordinates()
        self.point_movement_mode = 1
        self.ocr_centers = None

        self.get_ocr_centers()

    def get_xy_coordinates(self):
        xy_coordinates = list(zip(
            self.pandas_dataframe['CURRENT_FIX_X'], self.pandas_dataframe['CURRENT_FIX_Y']))
        xy_int_coordinates = [(int(x), int(y)) for x, y in xy_coordinates]
        self.fixation_coordinates = xy_int_coordinates

    def draw_points_on_image(self):
        # Draw all points on the image
        self.image = cv2.imread(self.image_path)

        for i, (x, y) in enumerate(self.fixation_coordinates):
            color = (0, 165, 255) if i == self.current_fixation_index else (
                (128, 0, 128))
            cv2.circle(self.image, (x, y), radius=10, color=color,
                       thickness=1)  # Circle for points
            if i < len(self.fixation_coordinates) - 1:
                next_x, next_y = self.fixation_coordinates[i + 1]
                cv2.line(self.image, (next_x, next_y),
                         (x, y), color=color, thickness=1)
        return self.image

    def move_point(self, direction):
        # Move the current active point in the specified direction
        x, y = self.fixation_coordinates[self.current_fixation_index]
        if self.point_movement_mode == 1:
            if direction == 'up':
                self.fixation_coordinates[self.current_fixation_index] = (
                    x, y - 10)  # Move up (decrease y)
            elif direction == 'down':
                self.fixation_coordinates[self.current_fixation_index] = (
                    x, y + 10)  # Move down (increase y)
            elif direction == 'left':
                self.fixation_coordinates[self.current_fixation_index] = (
                    x - 10, y)  # Move left (decrease x)
            elif direction == 'right':
                self.fixation_coordinates[self.current_fixation_index] = (
                    x + 10, y)  # Move right (increase x)
        elif self.point_movement_mode == 0:
            if direction == 'up':
                self.fixation_coordinates[self.current_fixation_index] = ocr_reader.find_closest_top_box(
                    x, y, self.ocr_centers)
            elif direction == 'down':
                self.fixation_coordinates[self.current_fixation_index] = ocr_reader.find_closest_bottom_box(
                    x, y, self.ocr_centers)
            elif direction == 'left':
                self.fixation_coordinates[self.current_fixation_index] = ocr_reader.find_closest_left_box(
                    x, y, self.ocr_centers)
            elif direction == 'right':
                self.fixation_coordinates[self.current_fixation_index] = ocr_reader.find_closest_right_box(
                    x, y, self.ocr_centers)

    def delete_fixation(self):
        self.last_deleted_fixation = self.fixation_coordinates[self.current_fixation_index]
        self.fixation_coordinates.pop(self.current_fixation_index)
        if self.row_to_be_deleted is not None:
            self.pandas_dataframe = self.pandas_dataframe.drop(
                self.pandas_dataframe.index[self.row_to_be_deleted])
        self.row_to_be_deleted = self.current_fixation_index

    def undo_last_fixation(self):
        if self.last_deleted_fixation is not None:
            self.fixation_coordinates.insert(
                self.current_fixation_index, self.last_deleted_fixation)
            self.last_deleted_fixation = None
            self.row_to_be_deleted = None

    def handle_key(self, key):
        if key == ord('w'):  # Move up
            self.move_point('up')
        elif key == ord('s'):  # Move down
            self.move_point('down')
        elif key == ord('a'):  # Move left
            self.move_point('left')
        elif key == ord('d'):  # Move right
            self.move_point('right')
        elif key == ord('n'):  # Go to the next point
            self.current_fixation_index += 1
            if self.current_fixation_index >= len(self.fixation_coordinates):
                self.current_fixation_index = 0  # Loop back to the first point
        elif key == ord('p'):  # Go to previous point
            self.current_fixation_index -= 1
        elif key == ord('q'):  # Exit editing
            if self.row_to_be_deleted is not None:
                self.pandas_dataframe = self.pandas_dataframe.drop(
                    self.pandas_dataframe.index[self.row_to_be_deleted])
            return False
        elif key == (ord('l')):
            self.delete_fixation()
        elif key == ord('u'):
            self.undo_last_fixation()
        elif key == ord('m'):
            self.switch_point_movement_mode()

        return True

    def edit_points(self):
        while self.current_fixation_index < len(self.fixation_coordinates):
            # Draw the points on the image
            image_with_points = self.draw_points_on_image()

            # Display the image with the overlaid points
            cv2.imshow(f'Page {self.image_path}', image_with_points)
            cv2.setWindowTitle(
                f'Page {self.image_path}', self.image_path[:-4] + ' ' + self.title)

            # Wait for a key press to move or select next point
            key = cv2.waitKey(0) & 0xFF  # Get key press

            if not self.handle_key(key):  # Handle the key press
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()

    def export_correct_fixation(self):
        # self.pandas_dataframe.to_csv(self.data_path, index=False)
        print(self.pandas_dataframe.shape)
        return self.pandas_dataframe

    def get_ocr_centers(self):
        reader = ocr_reader.OCR_Reader(self.image_path)
        reader.read_image()
        self.ocr_centers = reader.list_of_centers

    def switch_point_movement_mode(self):
        if self.point_movement_mode == 1:
            self.point_movement_mode = 0
        else:
            self.point_movement_mode = 1


class DataProcessing:
    def __init__(self, csv_file, x_column, y_column, image_column, session_column, image_folder):
        self.csv_file = csv_file
        self.dataframes = {}
        self.x_column = x_column
        self.y_column = y_column
        self.image_column = image_column
        self.session_column = session_column
        self.image_folder = image_folder
        self.image_list = os.listdir(self.image_folder)

    def prepare_data(self):
        raw_data = pd.read_csv(self.csv_file)
        # Drop the entries where there is no corresponding image
        dropped = raw_data[(raw_data[self.image_column] +
                            '.png').isin(self.image_list)]
        grouped_data = dropped.groupby(
            [self.image_column, self.session_column])
        for (image, session), group in grouped_data:
            self.dataframes[(image, session)] = group
        return self.dataframes

    def save_data(self, dataframe):
        dataframe.to_csv(self.csv_file, index=False)


def run_fixation_correction(csv_file, x_column, y_column, image_column, session_column, image_folder):
    prepared = DataProcessing(
        csv_file, x_column, y_column, image_column, session_column, image_folder)
    dataframes = prepared.prepare_data()

    corrected_dataframes = []
    cv2.waitKey(0) & 0xFF

    for entry in dataframes:
        df = dataframes[entry]
        image_name = dataframes[entry][image_column].iloc[0]
        for image in os.listdir(image_folder):
            if image_name in image:
                image_path = os.path.join(image_folder, image)
                fix = FixationCorrection(
                    image_path, df, df[session_column].iloc[0])
                fix.edit_points()
                corrected_dataframes.append(fix.pandas_dataframe)

    # save the corrected data in a new file
    if corrected_dataframes:
        combined_dataframe = pd.concat(corrected_dataframes)
        directory = os.path.dirname(csv_file)
        filename = os.path.basename(csv_file)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_fixation_corrected{ext}"
        new_path = os.path.join(directory, new_filename)
        combined_dataframe.to_csv(new_path, index=False)


run_fixation_correction('18sat_fixfinal.csv', 'CURRENT_FIX_X', 'CURRENT_FIX_Y',
                        'page_name', 'RECORDING_SESSION_LABEL', 'reading screenshot')

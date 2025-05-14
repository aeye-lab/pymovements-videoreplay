from __future__ import annotations
import os
import cv2
import ocr_reader
import pandas as pd
from pynput import keyboard


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
        self.correction = True
        self.original_fixation = None

        self.get_ocr_centers()
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

    def get_xy_coordinates(self):
        xy_coordinates = list(
            zip(
                self.pandas_dataframe['CURRENT_FIX_X'], self.pandas_dataframe['CURRENT_FIX_Y'],
            ),
        )
        xy_int_coordinates = [(int(x), int(y)) for x, y in xy_coordinates]
        self.fixation_coordinates = xy_int_coordinates

    def draw_points_on_image(self):
        # Draw all points on the image
        self.image = cv2.imread(self.image_path)

        for i, (x, y) in enumerate(self.fixation_coordinates):
            color = (0, 165, 255) if i == self.current_fixation_index else (
                (128, 0, 128)
            )
            cv2.circle(
                self.image, (x, y), radius=10, color=color,
                thickness=1,
            )  # Circle for points
            if i < len(self.fixation_coordinates) - 1:
                next_x, next_y = self.fixation_coordinates[i + 1]
                cv2.line(
                    self.image, (next_x, next_y),
                    (x, y), color=color, thickness=1,
                )
        return self.image

    def move_point(self, direction):
        # Move the current active point in the specified direction
        x, y = self.fixation_coordinates[self.current_fixation_index]

        if not self.original_fixation:
            self.original_fixation = (x,y)

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
                self.fixation_coordinates[self.current_fixation_index] = ocr_reader.find_closest_top_box(
                    x, y, self.ocr_centers,
                )
            elif direction == 'down':
                self.fixation_coordinates[self.current_fixation_index] = ocr_reader.find_closest_bottom_box(
                    x, y, self.ocr_centers,
                )
            elif direction == 'left':
                self.fixation_coordinates[self.current_fixation_index] = ocr_reader.find_closest_left_box(
                    x, y, self.ocr_centers,
                )
            elif direction == 'right':
                self.fixation_coordinates[self.current_fixation_index] = ocr_reader.find_closest_right_box(
                    x, y, self.ocr_centers,
                )

    def delete_fixation(self):
        self.last_deleted_fixation = self.fixation_coordinates[self.current_fixation_index]
        self.fixation_coordinates.pop(self.current_fixation_index)
        if self.row_to_be_deleted is not None:
            self.pandas_dataframe = self.pandas_dataframe.drop(
                self.pandas_dataframe.index[self.row_to_be_deleted],
            )
        self.row_to_be_deleted = self.current_fixation_index

    def undo_last_deletion(self):
        if self.last_deleted_fixation is not None:
            self.fixation_coordinates.insert(
                self.current_fixation_index, self.last_deleted_fixation,
            )
            self.last_deleted_fixation = None
            self.row_to_be_deleted = None

    def undo_last_correction(self):
        self.fixation_coordinates[self.current_fixation_index] = self.original_fixation

    def on_press(self, key):
        try:
            if key == keyboard.Key.up:
                self.move_point('up')
            elif key == keyboard.Key.down:
                self.move_point('down')
            elif key == keyboard.Key.left:
                self.current_fixation_index -= 1
                self.original_fixation = None
            elif key == keyboard.Key.right:
                self.current_fixation_index += 1
                self.original_fixation = None
                if self.current_fixation_index >= len(self.fixation_coordinates):
                    self.current_fixation_index = 0  # Loop back to the first point
            elif key.char == 'q':
                self.move_point('left')
            elif key.char == 'r':
                self.move_point('right')
            elif key.char == 'l':
                self.delete_fixation()
            elif key.char == 'u':
                self.undo_last_deletion()
            elif key.char == 'z':
                self.undo_last_correction()
            elif key.char == 'm':
                self.switch_point_movement_mode()
            elif key.char == 'n':
                if self.row_to_be_deleted is not None:
                    self.pandas_dataframe = self.pandas_dataframe.drop(
                        self.pandas_dataframe.index[self.row_to_be_deleted],
                    )
                self.save_corrected_fixations()
                self.correction = False

        except AttributeError:
            pass

    def edit_points(self):
        while self.current_fixation_index < len(self.fixation_coordinates):
            # Draw the points on the image
            image_with_points = self.draw_points_on_image()
            self.display_point_movement_mode()
            # Display the image with the overlaid points
            cv2.imshow(f'Page {self.image_path}', image_with_points)
            cv2.setWindowTitle(
                f'Page {self.image_path}', self.image_path[:-
                                                           4] + ' ' + self.title,
            )

            # Wait for a key press to move or select next point
            cv2.waitKey(0) & 0xFF  # Get key press

            if not self.correction:
                #self.save_corrected_fixations()
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()

    def save_corrected_fixations(self):
        self.pandas_dataframe[['x_corrected', 'y_corrected']] = pd.DataFrame(self.fixation_coordinates)

    def get_ocr_centers(self):
        reader = ocr_reader.OCR_Reader(self.image_path)
        reader.get_list_of_centers()
        self.ocr_centers = reader.list_of_centers

    def switch_point_movement_mode(self):
        if self.point_movement_mode == 1:
            try:
                self.point_movement_mode = 0
            except ModuleNotFoundError:
                print("Feature unavailable: 'pytesseract' is not installed.")
        else:
            self.point_movement_mode = 1

    def display_point_movement_mode(self):
        mode = ''
        if self.point_movement_mode == 0:
            mode = 'AOI'
        elif self.point_movement_mode == 1:
            mode = 'Pixel'

        # Coordinates for box
        box_top_left = (50, 10)
        box_bottom_right = (360, 60)

        # Draw the box
        cv2.rectangle(self.image, box_top_left, box_bottom_right,
                      (200, 200, 200), -1)  # Grey background
        cv2.rectangle(self.image, box_top_left, box_bottom_right,
                      (0, 0, 0), 1)  # Black border

        # Add text
        cv2.putText(
            self.image, 'Fixation movement mode (m): ' +
            mode, (box_top_left[0] + 10, box_top_left[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )


class DataProcessing:
    def __init__(self, csv_file, x_column, y_column, image_column, image_folder, groups=None, filters=None):
        self.csv_file = csv_file
        self.dataframes = {}
        self.x_column = x_column
        self.y_column = y_column
        self.image_column = image_column
        self.image_folder = image_folder
        self.image_list = os.listdir(self.image_folder)
        self.groups = groups
        self.filters = filters

        self.groups.append(self.image_column)


    def prepare_data(self):
        raw_data = pd.read_csv(self.csv_file)
        # Drop the entries where there is no corresponding image
        dropped = raw_data[(
            raw_data[self.image_column] +
            '.png'
        ).isin(self.image_list)]
        self.dataframes = self.filter_and_group(dropped)
        return self.dataframes

    def save_data(self, dataframe):
        dataframe.to_csv(self.csv_file, index=False)

    def filter_and_group(self, dataframe):
        if self.filters:
            for col, value in self.filters:
                if isinstance(value, list):
                    dataframe = dataframe[dataframe[col].isin(value)]
                else:
                    dataframe = dataframe[dataframe[col] == value]

        if self.groups:
            grouped = dataframe.groupby(self.groups)
            return [group.copy() for _, group in grouped]
        else:
            return [dataframe.copy()]



def run_fixation_correction(csv_file, x_column, y_column, image_column, image_folder, groups=None, filters=None):
    prepared = DataProcessing(
        csv_file, x_column, y_column, image_column, image_folder, groups, filters
    )
    dataframes = prepared.prepare_data()

    corrected_dataframes = []
    cv2.waitKey(0) & 0xFF

    for frame in dataframes:
        image_name = frame[image_column].iloc[0]
        for image in os.listdir(image_folder):
            if image_name in image:
                image_path = os.path.join(image_folder, image)
                fix = FixationCorrection(
                    image_path, frame, frame[groups[0]].iloc[0],
                )
                fix.edit_points()
                # fix.save_corrected_fixations()
                corrected_dataframes.append(fix.pandas_dataframe)



    # save the corrected data in a new file
    if corrected_dataframes:
        combined_dataframe = pd.concat(corrected_dataframes,ignore_index=True)
        directory = os.path.dirname(csv_file)
        filename = os.path.basename(csv_file)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_fixation_corrected{ext}"
        new_path = os.path.join(directory, new_filename)
        combined_dataframe.to_csv(new_path, index=False)


run_fixation_correction(
    '18sat_fixfinal.csv', 'CURRENT_FIX_X', 'CURRENT_FIX_Y',
    'page_name', 'reading screenshot',['RECORDING_SESSION_LABEL'],[('RECORDING_SESSION_LABEL','msd002')]
)

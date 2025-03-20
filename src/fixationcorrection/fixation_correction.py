import cv2
import pandas as pd
import os

class FixationCorrection:
    def __init__(self, image_path, pandas_dataframe, title=None):
        self.image_path = image_path
        self.pandas_dataframe = pandas_dataframe
        self.image = cv2.imread(self.image_path)
        self.current_fixation_index = 0  # Index of the current active circle
        self.fixation_coordinates = None  # Fixations for the current image
        self.title = title

        self.get_xy_coordinates()

    def get_xy_coordinates(self):
        xy_coordinates = list(zip(self.pandas_dataframe['CURRENT_FIX_X'], self.pandas_dataframe['CURRENT_FIX_Y']))
        xy_int_coordinates = [(int(x), int(y)) for x, y in xy_coordinates]
        self.fixation_coordinates = xy_int_coordinates

    def draw_points_on_image(self):
        # Draw all points on the image
        self.image = cv2.imread(self.image_path)

        for i, (x, y) in enumerate(self.fixation_coordinates):
            color = (255, 255, 0) if i == self.current_fixation_index else (0, 0, 255)
            cv2.circle(self.image, (x, y), radius=10, color=color, thickness=1)# Circle for points
            if i < len(self.fixation_coordinates) - 1:
                next_x, next_y = self.fixation_coordinates[i + 1]
                cv2.line(self.image, (next_x, next_y), (x, y), color=color, thickness=1)
        return self.image

    def move_point(self, direction):
        # Move the current active point in the specified direction
        x, y = self.fixation_coordinates[self.current_fixation_index]
        if direction == 'up':
            self.fixation_coordinates[self.current_fixation_index] = (x, y - 10)  # Move up (decrease y)
        elif direction == 'down':
            self.fixation_coordinates[self.current_fixation_index] = (x, y + 10)  # Move down (increase y)
        elif direction == 'left':
            self.fixation_coordinates[self.current_fixation_index] = (x - 10, y)  # Move left (decrease x)
        elif direction == 'right':
            self.fixation_coordinates[self.current_fixation_index] = (x + 10, y)  # Move right (increase x)


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
        elif key == ord('p'): #Go to previous point
            self.current_fixation_index -= 1
        elif key == ord('q'):  # Exit editing (optional)
            return False
        return True

    def edit_points(self):
        while self.current_fixation_index < len(self.fixation_coordinates):
            # Draw the points on the image
            image_with_points = self.draw_points_on_image()

            # Display the image with the overlaid points
            cv2.imshow(f'Page {self.image_path}', image_with_points)
            cv2.setWindowTitle(f'Page {self.image_path}', self.image_path[:-4] + ' ' + self.title)

            # Wait for a key press to move or select next point
            key = cv2.waitKey(0) & 0xFF  # Get key press

            if not self.handle_key(key):  # Handle the key press
                cv2.destroyAllWindows()
                return


        cv2.destroyAllWindows()




class DataPreparation:
    def __init__(self, csv_file, image_row_name, session_row_name, image_folder):
        self.csv_file = csv_file
        self.dataframes = {}
        self.image_row_name = image_row_name
        self.session_row_name = session_row_name
        self.image_folder = image_folder
        self.image_list = os.listdir(self.image_folder)

    def prepareData(self):
            raw_data = pd.read_csv(self.csv_file)
            # Drop the entries where there is no corresponding image
            dropped = raw_data[(raw_data['page_name'] + '.png').isin(self.image_list)]
            grouped_data = dropped.groupby([self.image_row_name, self.session_row_name])
            for (image, session), group in grouped_data:
                self.dataframes[(image, session)] = group
            return self.dataframes




prepared = DataPreparation('18sat_fixfinal.csv', 'page_name', 'RECORDING_SESSION_LABEL', 'reading screenshot')
prep = prepared.prepareData()

for entry in prep:
    df = prep[entry]
    image_name = prep[entry]['page_name'].iloc[0]
    fix = FixationCorrection(image_name + '.png', df, df['RECORDING_SESSION_LABEL'].iloc[0])
    fix.edit_points()
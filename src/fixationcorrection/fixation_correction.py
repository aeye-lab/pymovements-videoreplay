import cv2
import pandas as pd
import numpy as np


class FixationCorrection:
    def __init__(self, csv_file, window_width=800, window_height=600, step_size=10):
        # Initialize the class with file path and window dimensions
        self.data = None
        self.csv_file = csv_file
        self.window_width = window_width
        self.window_height = window_height
        self.step_size = step_size
        self.coordinates_updated = False
        self.current_index = 1

        # Load the CSV data
        self.load_data()

        # Set the initial position of the circle
        self.current_x = round(self.x_coords[self.current_index])
        self.current_y = round(self.y_coords[self.current_index])

    def load_data(self):
        # Load the CSV file and process coordinates
        self.data = pd.read_csv(self.csv_file)
        self.data['CURRENT_FIX_X'] = self.data['CURRENT_FIX_X'].apply(self.convert_comma_to_float)
        self.data['CURRENT_FIX_Y'] = self.data['CURRENT_FIX_Y'].apply(self.convert_comma_to_float)

        # Extract the necessary columns
        self.timestamps = self.data['CURRENT_FIX_INDEX']
        self.x_coords = self.data['CURRENT_FIX_X']
        self.y_coords = self.data['CURRENT_FIX_X']

    @staticmethod
    def convert_comma_to_float(value):
        """Convert values with comma to float."""
        return float(value.replace(',', '.'))

    def draw_circle(self):
        """Draw the circle and coordinates on the window."""
        img = 255 * np.ones(shape=[self.window_height, self.window_width, 3], dtype=np.uint8)

        # Draw the circle at the current position
        radius = 20
        color = (0, 0, 255)  # Red in BGR
        thickness = -1  # Filled circle
        cv2.circle(img, (self.current_x, self.current_y), radius, color, thickness)

        # Show the image
        cv2.imshow("FixationCorrection", img)

    def move_circle(self, key):
        """Move the circle based on the key pressed."""
        if key == ord('d'):  # 'D' key to move right
            self.current_x += self.step_size
            self.coordinates_updated = True
        elif key == ord('a'):  # 'A' key to move left
            self.current_x -= self.step_size
            self.coordinates_updated = True
        elif key == ord('s'):  # 'S' key to move down
            self.current_y += self.step_size
            self.coordinates_updated = True
        elif key == ord('w'):  # 'W' key to move up
            self.current_y -= self.step_size
            self.coordinates_updated = True

    def save_coordinates(self):
        """Save the updated coordinates."""
        self.x_coords[self.current_index] = self.current_x
        self.y_coords[self.current_index] = self.current_y
        print(
            f"Coordinates for timestamp {self.timestamps[self.current_index]} saved: ({self.current_x}, {self.current_y})")
        self.coordinates_updated = False  # Reset the flag

    def iterate_timestamps(self, key):
        """Iterate through the timestamps based on key presses."""
        if key == ord('n'):  # Right arrow key
            if self.current_index < len(self.timestamps) - 1:
                self.current_index += 1
                self.current_x = round(self.x_coords[self.current_index])
                self.current_y = round(self.y_coords[self.current_index])
            self.draw_circle()

        elif key == ord('p'):  # Left arrow key
            if self.current_index > 0:
                self.current_index -= 1
                self.current_x = round(self.x_coords[self.current_index])
                self.current_y = round(self.y_coords[self.current_index])
            self.draw_circle()

    def run(self):
        """Start the interactive window and handle key events."""
        while True:
            key = cv2.waitKey(0)  # Wait for a key press

            # Move the circle using the arrow or WASD keys
            self.move_circle(key)

            # Save coordinates with 's' key
            if key == ord('k') and self.coordinates_updated:
                self.save_coordinates()

            # Navigate through timestamps with left/right arrow keys
            self.iterate_timestamps(key)

            # Press 'q' to exit the loop
            if key == ord('q'):
                break

            # Redraw the circle after each key press
            self.draw_circle()

        # Release OpenCV windows
        cv2.destroyAllWindows()

    def save_data(self, output_file="updated_coordinates.csv"):
        """Save the updated coordinates to a CSV file."""
        self.data['CURRENT_FIX_X'] = self.x_coords
        self.data['CURRENT_FIX_Y'] = self.y_coords
        self.data.to_csv(output_file, index=False)


# Example of how to use the class:
if __name__ == "__main__":
    editor = FixationCorrection('data.csv')  # Replace with your CSV file path
    editor.run()

    # Optionally, save the updated coordinates to a new CSV
   # editor.save_data("updated_coordinates.csv")

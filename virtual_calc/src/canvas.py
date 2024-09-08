import numpy as np
import cv2

class Canvas:
    def __init__(self):
        self.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.result_text = ""

    def draw(self, position):
        cv2.circle(self.canvas, position, 5, (255, 0, 0), -1)

    def navigate(self, position):
        # Add logic to navigate the canvas
        pass

    def reset(self):
        self.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.result_text = ""

    def get_image(self):
        return self.canvas

    def display_result(self, result):
        self.result_text = result

    def get_combined_frame(self, frame):
        combined = cv2.addWeighted(frame, 0.7, self.canvas, 0.3, 0)
        return combined

    def get_result_text(self):
        return self.result_text

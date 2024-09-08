import numpy as np
import cv2

class Canvas:
    def __init__(self):
        self.canvas = None

    def initialize_canvas(self, frame):
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

    def draw_on_canvas(self, hand_landmarks):
        if hand_landmarks:
            for landmarks in hand_landmarks:
                index_finger_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * self.canvas.shape[1])
                y = int(index_finger_tip.y * self.canvas.shape[0])
                cv2.circle(self.canvas, (x, y), 5, (255, 0, 0), -1)

    def get_combined_frame(self, frame):
        return cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)

    def get_canvas(self):
        return self.canvas

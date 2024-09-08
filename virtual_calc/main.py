import cv2
import numpy as np
from src.hand_detection import HandDetector
from src.canvas import Canvas
from models.google_api import GeminiAPI
from config.settings import API_KEY

def main():
    hand_detector = HandDetector()
    gemini_api = GeminiAPI(api_key=API_KEY)
    canvas = Canvas()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Virtual Calculator', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Virtual Calculator', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame to fix the mirrored view
        frame = cv2.flip(frame, 1)

        # Detect hand gestures
        frame = hand_detector.detect(frame)
        gestures = hand_detector.get_gestures()

        # Perform actions based on gestures
        if gestures == 'index_finger_up':
            finger_tip_position = hand_detector.get_finger_tip_position()
            if finger_tip_position:
                canvas.draw(finger_tip_position)
        elif gestures == 'two_fingers_up':
            hand_position = hand_detector.get_hand_position()
            if hand_position:
                canvas.navigate(hand_position)
        elif gestures == 'thumb_up':
            canvas.reset()
        elif gestures == 'small_finger_up':
            result = gemini_api.solve(canvas.get_image())
            canvas.display_result(result)

        combined_frame = canvas.get_combined_frame(frame)  # Pass frame here

        # Create a white space for displaying the answer on the right side
        answer_area = np.zeros((combined_frame.shape[0], 300, 3), dtype=np.uint8) + 255
        combined_frame = np.hstack((combined_frame, answer_area))

        # Show the result in the answer area
        result = canvas.get_result_text()
        y0, dy = 30, 20  # Adjust dy for smaller font size
        for i, line in enumerate(wrap_text(result, 25)):  # Adjust max_width for smaller font size
            y = y0 + i * dy
            cv2.putText(combined_frame, line, (combined_frame.shape[1] - 290, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow('Virtual Calculator', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def wrap_text(text, max_width):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= max_width:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines

if __name__ == '__main__':
    main()

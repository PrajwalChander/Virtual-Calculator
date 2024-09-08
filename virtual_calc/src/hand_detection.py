import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, max_hands=1):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=max_hands)
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def get_gestures(self):
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                fingers = []

                # Thumb
                if landmarks[4].x < landmarks[3].x:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Other four fingers
                for id in range(8, 21, 4):
                    if landmarks[id].y < landmarks[id - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                if fingers == [0, 1, 0, 0, 0]:
                    return 'index_finger_up'
                elif fingers == [0, 1, 1, 0, 0]:
                    return 'two_fingers_up'
                elif fingers == [1, 0, 0, 0, 0]:
                    return 'thumb_up'
                elif fingers == [0, 0, 0, 0, 1]:
                    return 'small_finger_up'
                else:
                    return 'unknown'
        return None

    def get_finger_tip_position(self):
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Index finger tip is landmark 8
                x = int(hand_landmarks.landmark[8].x * 640)
                y = int(hand_landmarks.landmark[8].y * 480)
                return (x, y)
        return None

    def get_hand_position(self):
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[0].x * 640)
                y = int(hand_landmarks.landmark[0].y * 480)
                return (x, y)
        return None

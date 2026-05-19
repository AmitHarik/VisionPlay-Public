import cv2
import mediapipe as mp
import csv
import json
import time
class GestCollect:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)
        
        # press num key while holding gesture to save 
        try:
            with open("gestures.json", "r") as f:
                self.keys = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("gestures.json missing or invalid, using defaults")
            self.keys = {
                "fist": "play", "open_palm": "pause", "index_up": "volume_up",
                "two_up": "volume_down", "thumb_right": "next_track", "thumb_left": "prev_track"
            }
            
        self.gesture_names = list(self.keys.keys())
        self.counts = {g: 0 for g in self.gesture_names}
        
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def proc_lms(self, lms):
        # normalise to wrist then scale to -1 1 
        w = lms.landmark[0]
        coords = []
        for lm in lms.landmark:
            coords.extend([lm.x - w.x, lm.y - w.y, lm.z - w.z])
        mv = max(abs(c) for c in coords)
        if mv > 0:
            coords = [c/mv for c in coords]
        return coords

    def run(self):
        while True:
            ret, frame = self.cam.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            if res.multi_hand_landmarks:
                for lms in res.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)

                    k = cv2.waitKey(1)
                    if k != -1 and k != ord('q'):
                        # map number keys 
                        idx = k - ord('1')
                        if 0 <= idx < len(self.gesture_names):
                            label = self.gesture_names[idx]
                            self.counts[label] += 1

                            coords = self.proc_lms(lms)

                            with open("gesture_data.csv", "a", newline='') as f:
                                csv.writer(f).writerow([label] + coords)
                            print(f"saved {label} #{self.counts[label]}")

            cv2.imshow("collect", frame)
            if cv2.waitKey(1) == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    gc = GestCollect()
    gc.run()
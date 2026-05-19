import collections
import cv2
import mediapipe as mp
import time
import joblib
import json

# proc - process
# perf - performance
# gest - gesture
# conf - confidence
# inf - inference
# up - update

class GestureProc:

    RED=(0,0,255) 
    GREEN=(0,255,0); 
    BLUE=(255,0,0)
    WHITE=(255,255,255)
    YELLOW=(0,255,255)
    CYAN=(255,255,0)
    MAGENTA=(255,0,255)
    BLACK=(0,0,0)

    def __init__(self, controller=None):
        self.controller = controller

        try:
            with open("gestures.json", "r") as f:
                self.gest_map = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("gestures.json missing or invalid, using defaults")
            self.gest_map = {
                "fist": "play", "open_palm": "pause", "index_up": "volume_up",
                "two_up": "volume_down", "thumb_right": "next_track", "thumb_left": "prev_track"
            }

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_style = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1, 
            model_complexity=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.6)

        self.lm_style = None
        self.cn_style = None
        self.last_gest = ""
        self.last_t = 0.0
        self.use_ml = False # machine learning
        self.cooldown = 1.5  # secs between actions
        self.buf = collections.deque(maxlen=3)  # need 3 same frames to fire
        self.is_playing = None  # spotify state none means idk yet
        self.last_conf = 0.0
        self.last_inf_ms = 0.0

        try:
            self.model = joblib.load("gesture_model_cv.pkl")
            print("loaded ml model")
        except FileNotFoundError:
            print("no pkl found, heuristic only")
            self.model = None

    def up_style(self, lm_colour, cn_colour):
        cmap = {
            "red": self.RED, "green": self.GREEN, "blue": self.BLUE,
            "white": self.WHITE, "yellow": self.YELLOW,
            "cyan": self.CYAN, "magenta": self.MAGENTA, "black": self.BLACK,
        }
        if lm_colour == "default":
            self.lm_style = None
        else:
            self.lm_style = self.mp_draw.DrawingSpec(color=cmap.get(lm_colour, self.RED), thickness=2, circle_radius=2)

        if cn_colour == "default":
            self.cn_style = None
        else:
            self.cn_style = self.mp_draw.DrawingSpec(color=cmap.get(cn_colour, self.GREEN), thickness=2)

    def proc_frame(self, bgr):
        # flip is in app.py 
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # slight perf boost
        res = self.hands.process(rgb)
        rgb.flags.writeable = True
        ann = bgr.copy()
        g = None

        if res.multi_hand_landmarks:
            for lms in res.multi_hand_landmarks:
                lm_s = self.lm_style or self.mp_style.get_default_hand_landmarks_style()
                cn_s = self.cn_style or self.mp_style.get_default_hand_connections_style()
                self.mp_draw.draw_landmarks(ann, lms, self.mp_hands.HAND_CONNECTIONS, lm_s, cn_s)

                g = self._get_gest(lms.landmark)
                self.buf.append(g)

                # only fire if last 3 frames same 
                if len(self.buf) == 3 and len(set(self.buf)) == 1:
                    if g:
                        self.fire(g)
                    else:
                        self.last_gest = None
        else:
            self.buf.clear()
            self.last_gest = None
        return g, ann

    def switch_mode(self, ml_on):
        if ml_on and not self.model:
            return False
        self.use_ml = ml_on
        return True

    def proc_lms(self, lms):
        # normalise to wrist then scale to -1 1
        w = lms[0]
        coords = []
        for lm in lms:
            coords.extend([lm.x - w.x, lm.y - w.y, lm.z - w.z])
        mv = max(abs(c) for c in coords)
        if mv > 0:
            coords = [c/mv for c in coords]
        return coords

    def _get_gest(self, lms):
        if self.use_ml:
            coords = self.proc_lms(lms)

            t0 = time.perf_counter()
            proba = self.model.predict_proba([coords])[0]
            self.last_inf_ms = (time.perf_counter() - t0) * 1000

            self.last_conf = max(proba)
            if self.last_conf < 0.7:
                return None  # not confident
            return self.model.classes_[proba.argmax()]

        # tip y < mcp y means finger extended
        iu = lms[8].y < lms[5].y   # index
        mu = lms[12].y < lms[9].y   # middle
        ru = lms[16].y < lms[13].y  # ring
        pu = lms[20].y < lms[17].y  # pinky

        if not any([iu, mu, ru, pu]):
            dx = lms[4].x - lms[2].x
            dy = abs(lms[4].y - lms[2].y)
            ext = abs(lms[4].x - lms[5].x)
            # thumb away from index
            if abs(dx) > dy*1.2 and abs(dx) > 0.1 and ext > 0.08:
                return "thumb_right" if dx > 0 else "thumb_left"
            return "fist"

        if iu and not mu and not ru and not pu:
            return "index_up"
        if iu and mu and not ru and not pu:
            return "two_up"
        if iu and mu and ru and pu:
            return "open_palm"
        return None  # ignore partial gestures

    def upd_cooldown(self, val):
        self.cooldown = float(val)

    def fire(self, g):
        t = time.time()

        if g == self.last_gest:
            return  # gest still held dont repeat
        if (t - self.last_t) < self.cooldown:
            return  # too fast

        action = self.gest_map.get(g)

        # skip if spotify already doing it 
        if self.is_playing is not None:
            if action == "play" and self.is_playing:
                self.last_gest = g
                self.last_t = t
                return
            if action == "pause" and not self.is_playing:
                self.last_gest = g
                self.last_t = t
                return

        self.last_gest = g
        self.last_t = t

        if action and self.controller:
            fn = getattr(self.controller, action, None)
            if callable(fn):
                print(f"{action} triggered")
                fn()
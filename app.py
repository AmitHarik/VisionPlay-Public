import cv2
import customtkinter as ctk
import csv
from PIL import Image
import requests
from io import BytesIO
import time
from main import MediaController, SpotifyController
from gesturerecog import GestureProc
from pathlib import Path
import queue
import threading

# palette consts
BG = "#1a1a1a"
SURF = "#242424"
SURF_2 = "#2e2e2e"
TXT_MAIN = "#f0f0f0"
TXT_MUTE = "#888888"
ACCENT = "#1DB954" #spotify green

# fonts
F_SM = ("Roboto", 12)
F_LG = ("Roboto", 15)
F_LG_BOLD = ("Roboto", 16, "bold")

# config
WIN_SIZE = "1150x780"
CAM_W = 640
CAM_H = 480
POLL_CAM = 30
POLL_SPOTIFY = 1000
MAX_LOG = 10
ART_SIZE = 96
PBAR_H = 8

class VisionPlayApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VisionPlay")
        self.geometry(WIN_SIZE)
        self.configure(fg_color=BG)
        ctk.set_appearance_mode("dark")

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.mc = MediaController()
        self.sp = SpotifyController()
        self.detector = GestureProc(controller=self.mc)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, CAM_W)
        self.cap.set(4, CAM_H)

        self.log_g = []
        self.curr_img = None  # keep last art url so no redownload
        self.last_gest = "None"

        self.perf_log = open("perf_metrics.csv", "w", newline="")
        self.perf_writer = csv.writer(self.perf_log)
        self.perf_writer.writerow(["time_elapsed", "mode", "fps", "inference_ms", "confidence"])
        self.start_time = time.time()

        self.setup_gui()
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self.run_cam()
        self.run_spotify()

    def _log(self, msg):
        self.log_g.append(msg)
        if len(self.log_g) > MAX_LOG:
            self.log_g.pop(0)
        self.llbl.configure(text="\n".join(self.log_g))


    def setup_gui(self):
        top_bar = ctk.CTkFrame(self, fg_color=SURF, corner_radius=0, height=64)
        top_bar.grid(row=0, column=0, sticky="ew")
        top_bar.grid_columnconfigure(2, weight=1)

        tbox = ctk.CTkFrame(top_bar, fg_color="transparent")
        tbox.grid(row=0, column=0, padx=24, pady=16, sticky="w")
        ctk.CTkLabel(tbox, text="VisionPlay.", font=F_LG_BOLD, text_color=ACCENT).pack(side="left")
        ctk.CTkLabel(tbox, text="Vision-Based Media Controller", font=F_SM, text_color=TXT_MUTE).pack(side="left", padx=(16, 0), pady=(2, 0))

        pf = ctk.CTkFrame(top_bar, fg_color="transparent")
        pf.grid(row=0, column=2, sticky="e", padx=24)

        def make_pill(parent, text):
            lbl = ctk.CTkLabel(parent, text=text, font=F_SM, text_color=TXT_MUTE,
                               fg_color=SURF_2, corner_radius=16, padx=16, pady=4)
            lbl.pack(side="left", padx=8)
            return lbl

        self.c_status = make_pill(pf, "initialising camera")
        self.s_status = make_pill(pf, "connecting to spotify")

        mf = ctk.CTkFrame(self, fg_color="transparent")
        mf.grid(row=1, column=0, sticky="nsew", padx=24, pady=24)
        mf.grid_rowconfigure(0, weight=1)
        mf.grid_columnconfigure(0, weight=6)
        mf.grid_columnconfigure(1, weight=4)

        left = ctk.CTkFrame(mf, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        cam_box = ctk.CTkFrame(left, fg_color=SURF, corner_radius=16)
        cam_box.grid(row=0, column=0, sticky="nsew")
        cam_box.grid_rowconfigure(0, weight=1)
        cam_box.grid_columnconfigure(0, weight=1)

        self.vid_lbl = ctk.CTkLabel(cam_box, text="connecting to camera feed", font=F_LG, text_color=TXT_MUTE)
        self.vid_lbl.grid(row=0, column=0, sticky="nsew", padx=24, pady=24)

        over = ctk.CTkFrame(cam_box, fg_color=SURF_2, corner_radius=10)
        over.grid(row=0, column=0, sticky="se", padx=24, pady=24)

        self.gest_lbl = ctk.CTkLabel(over, text="Looking for gestures", font=F_SM, text_color=TXT_MUTE)
        self.gest_lbl.pack(anchor="w", padx=16, pady=(12, 2))
        self.cmd_lbl = ctk.CTkLabel(over, text="Waiting for commands", font=F_SM, text_color=TXT_MUTE)
        self.cmd_lbl.pack(anchor="w", padx=16, pady=(2, 12))

        self.conf_lbl = ctk.CTkLabel(over, text="Confidence:", font=F_SM, text_color=TXT_MUTE)
        
        self.perf_lbl = ctk.CTkLabel(over, text="fps: -  inf: -", font=F_SM, text_color=TXT_MUTE)
        self.perf_lbl.pack(anchor="w", padx=16, pady=(0, 12))

        bl = ctk.CTkFrame(left, fg_color="transparent")
        bl.grid(row=1, column=0, sticky="ew", pady=(16, 0))
        leg = ctk.CTkFrame(bl, fg_color=SURF, corner_radius=16)
        leg.pack(fill="x")
        leg_str = " | ".join(f"{k.replace('_', ' ')} = {v.replace('_', ' ')}" for k, v in self.detector.gest_map.items())
        ctk.CTkLabel(
            leg,
            text=leg_str,
            font=F_SM, text_color=TXT_MUTE, pady=16
        ).pack()

        right = ctk.CTkScrollableFrame(mf, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(16, 0))

        scard = ctk.CTkFrame(right, fg_color=SURF, corner_radius=16)
        scard.pack(fill="x", pady=(0, 16))

        self.art = ctk.CTkLabel(scard, text="no art", font=F_SM, text_color=TXT_MUTE, fg_color=SURF_2, corner_radius=8, width=ART_SIZE, height=ART_SIZE)
        self.art.pack(pady=(24, 16))

        self.s_name = ctk.CTkLabel(scard, text="no active playback", font=F_LG_BOLD, text_color=TXT_MAIN)
        self.s_name.pack(pady=(0, 4))

        self.s_artist = ctk.CTkLabel(scard, text="waiting for spotify connection", font=F_SM, text_color=TXT_MUTE)
        self.s_artist.pack(pady=(0, 4))

        self.s_device = ctk.CTkLabel(scard, text="no active device", font=F_SM, text_color=TXT_MUTE)
        self.s_device.pack(pady=(0, 16))

        self.pbar = ctk.CTkProgressBar(scard, width=280, height=PBAR_H, progress_color=ACCENT, fg_color=SURF_2)
        self.pbar.set(0.0)
        self.pbar.pack(padx=24, pady=(0, 24))

        self.qframe = ctk.CTkFrame(scard, fg_color=SURF_2, corner_radius=8)
        self.qframe.pack(fill="x", padx=16, pady=(0, 16))
        ctk.CTkLabel(self.qframe, text="up next", font=F_LG_BOLD, text_color=TXT_MAIN).pack(anchor="w", padx=16, pady=(8, 4))

        self.qlabels = []
        for i in range(4):
            lbl = ctk.CTkLabel(self.qframe, text="", font=F_SM, text_color=TXT_MUTE, anchor="w")
            lbl.pack(fill="x", padx=16, pady=2)
            self.qlabels.append(lbl)

        ccard = ctk.CTkFrame(right, fg_color=SURF, corner_radius=16)
        ccard.pack(fill="x", pady=(0, 16), ipady=8)

        #getting icons
        def get_ico(n, sz=24):
            p = Path(__file__).parent / "icons" / f"{n}.png"
            return ctk.CTkImage(light_image=Image.open(p).convert("RGBA"), size=(sz, sz)) if p.exists() else None

        ico_prev = get_ico("skip_previous")
        ico_play = get_ico("play_pause", 32)
        ico_next = get_ico("skip_next")
        ico_vold = get_ico("volume_down")
        ico_volu = get_ico("volume_up")

        b_sty = {"width": 64, "height": 48, "fg_color": SURF_2, "hover_color": "#383838", "text_color": TXT_MAIN, "corner_radius": 8, "cursor": "hand2"}
        ctk.CTkButton(ccard, image=ico_prev, text="" if ico_prev else "prev", font=F_LG, command=self.mc.prev_track, **b_sty).pack(side="left", expand=True, padx=8, pady=8)
        ctk.CTkButton(ccard, image=ico_play, text="" if ico_play else "play/pause", font=F_LG, command=self.mc.play, **b_sty).pack(side="left", expand=True, padx=8, pady=8)
        ctk.CTkButton(ccard, image=ico_next, text="" if ico_next else "next", font=F_LG, command=self.mc.next_track, **b_sty).pack(side="left", expand=True, padx=8, pady=8)
        ctk.CTkButton(ccard, image=ico_vold, text="" if ico_vold else "vol down", font=F_LG, command=self.mc.volume_down, **b_sty).pack(side="left", expand=True, padx=8, pady=8)
        ctk.CTkButton(ccard, image=ico_volu, text="" if ico_volu else "vol up", font=F_LG, command=self.mc.volume_up, **b_sty).pack(side="left", expand=True, padx=8, pady=8)

        set_f = ctk.CTkFrame(right, fg_color="transparent")
        set_f.pack(fill="x", pady=(0, 16))
        set_f.grid_columnconfigure(0, weight=1)
        set_f.grid_columnconfigure(1, weight=1)

        clcard = ctk.CTkFrame(set_f, fg_color=SURF, corner_radius=16)
        clcard.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ctk.CTkLabel(clcard, text="classifier", font=F_LG_BOLD, text_color=TXT_MAIN).pack(pady=(16, 8))
        self.m_var = ctk.StringVar(value="heuristic")
        self.sw = ctk.CTkSegmentedButton(clcard, values=["heuristic", "ml"],
                                         variable=self.m_var, command=self.toggle_mode,
                                         selected_color=ACCENT, unselected_color=SURF_2,
                                         selected_hover_color=ACCENT, unselected_hover_color="#383838",
                                         text_color=TXT_MAIN, font=F_SM)
        self.sw.pack(padx=16, pady=8)
        self.ml_warn = ctk.CTkLabel(clcard, text="", font=F_SM, text_color=TXT_MUTE)
        self.ml_warn.pack(pady=(0, 16))

        cdcard = ctk.CTkFrame(set_f, fg_color=SURF, corner_radius=16)
        cdcard.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ctk.CTkLabel(cdcard, text="cooldown", font=F_LG_BOLD, text_color=TXT_MAIN).pack(pady=(16, 0))
        self.cd_txt = ctk.CTkLabel(cdcard, text=f"{self.detector.cooldown:.1f}s", font=F_LG, text_color=TXT_MAIN)
        self.cd_txt.pack()
        sldr = ctk.CTkSlider(cdcard, from_=0.5, to=3.0, number_of_steps=25,
                             button_color=ACCENT, button_hover_color=ACCENT, progress_color=ACCENT, fg_color=SURF_2, command=self.change_cd)
        sldr.set(self.detector.cooldown)
        sldr.pack(padx=24, pady=(8, 16))

        bot_row = ctk.CTkFrame(right, fg_color="transparent")
        bot_row.pack(fill="both", expand=True, pady=(0, 16))
        bot_row.grid_columnconfigure(0, weight=1)
        bot_row.grid_columnconfigure(1, weight=1)
        bot_row.grid_rowconfigure(0, weight=1)

        style_card = ctk.CTkFrame(bot_row, fg_color=SURF, corner_radius=16)
        style_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ctk.CTkLabel(style_card, text="styles", font=F_LG_BOLD, text_color=TXT_MAIN).pack(pady=(16, 8))
        colours = ["default", "red", "green", "blue", "white", "yellow", "cyan", "magenta", "black"]
        self.lm_col = ctk.StringVar(value="default")
        self.cn_col = ctk.StringVar(value="default")

        om_sty = {
            "fg_color": SURF_2, "button_color": SURF_2, "button_hover_color": "#383838",
            "dropdown_fg_color": SURF, "dropdown_hover_color": SURF_2,
            "dropdown_text_color": TXT_MAIN, "text_color": TXT_MAIN, "font": F_SM
        }
        ctk.CTkLabel(style_card, text="landmarks", font=F_SM, text_color=TXT_MUTE).pack()
        ctk.CTkOptionMenu(style_card, values=colours, variable=self.lm_col, command=self.change_style, **om_sty).pack(padx=16, pady=(4, 8))
        ctk.CTkLabel(style_card, text="connectors", font=F_SM, text_color=TXT_MUTE).pack()
        ctk.CTkOptionMenu(style_card, values=colours, variable=self.cn_col, command=self.change_style, **om_sty).pack(padx=16, pady=(4, 16))

        log_box = ctk.CTkFrame(bot_row, fg_color=SURF, corner_radius=16)
        log_box.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ctk.CTkLabel(log_box, text="log", font=F_LG_BOLD, text_color=TXT_MAIN).pack(anchor="w", padx=24, pady=(16, 8))

        self.llbl = ctk.CTkLabel(log_box, text="system up.", font=F_SM, text_color=TXT_MUTE, justify="left")
        self.llbl.pack(anchor="nw", padx=24, pady=(0, 16))

    def change_style(self, _=None):
        self.detector.up_style(self.lm_col.get(), self.cn_col.get())

    def run_cam(self):
        t0 = time.perf_counter()
        ret, frm = self.cap.read()
        if ret:
            self.c_status.configure(text="camera active", text_color=TXT_MAIN)
            frm = cv2.flip(frm, 1)  # mirrored for natural use
            g, ann = self.detector.proc_frame(frm)

            if g:
                if g != self.last_gest:
                    self.last_gest = g
                    cmd = self.detector.gest_map.get(g, "None")
                    self.gest_lbl.configure(text=f"recognised: {g}", text_color=TXT_MAIN)
                    self.cmd_lbl.configure(text=f"command: {cmd}", text_color=TXT_MAIN)
                    self._log(f"{g} fired {cmd}")
            else:
                self.gest_lbl.configure(text="listening for gestures", text_color=TXT_MUTE)
                self.cmd_lbl.configure(text="standing by", text_color=TXT_MUTE)

            fps = 1.0 / (time.perf_counter() - t0) #fps counter
            elapsed = round(time.time() - self.start_time, 2) #time sicne app opened
            if self.detector.use_ml:
                pct = self.detector.last_conf * 100
                self.conf_lbl.configure(text=f"confidence: {pct:.0f}%", text_color=TXT_MAIN)
                self.conf_lbl.pack(anchor="w", padx=16, pady=(2, 2))
                self.perf_lbl.configure(text=f"fps: {fps:.1f} | inf: {self.detector.last_inf_ms:.1f}ms")
                self.perf_writer.writerow([elapsed, "ml", round(fps, 1), round(self.detector.last_inf_ms, 1), round(pct, 1)])
            else:
                self.conf_lbl.pack_forget()  # only needed in ml mode
                self.perf_lbl.configure(text=f"fps: {fps:.1f} | inf: n/a")
                self.perf_writer.writerow([elapsed, "heuristic", round(fps, 1), 0.0, 0.0])
                
            self.perf_log.flush()

            # bgr to rgb for pillow
        try:
            i = Image.fromarray(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
            i = i.resize((CAM_W, CAM_H))
            self.photo = ctk.CTkImage(light_image=i, size=(CAM_W, CAM_H))
            self.vid_lbl.configure(image=self.photo, text="")
        except Exception as e:
            self.vid_lbl.configure(text="could not display feed", text_color=TXT_MUTE)
            print(e)

        # keep the loop going
        self.after(POLL_CAM, self.run_cam)

    def run_spotify(self):
        # run fetch in background to avoid blocking main thread
        threading.Thread(target=self._fetch_spotify_data, daemon=True).start()
        self.after(POLL_SPOTIFY, self.run_spotify)

    def _fetch_spotify_data(self):
        tr = self.sp.get_song()
        dev = None
        queue_data = []
        url = None
        img_data = None

        if tr:
            dev = self.sp.get_dev()
            queue_data = self.sp.get_queue()
            url = tr.get("image_url")

            if url and url != self.curr_img:
                # only get art if track changed
                try:
                    res = requests.get(url)
                    img_data = res.content
                except requests.exceptions.ConnectionError:
                    print("network dropped, couldn't grab spotify art")
                    img_data = "error"
                except requests.exceptions.Timeout:
                    print("spotify cdn timed out")
                    img_data = "error"
                except Exception as e:
                    print(f"random os error loading art: {e}")
                    img_data = "error"

        self.after(0, self._update_spotify_ui, tr, dev, queue_data, url, img_data)

    def _update_spotify_ui(self, tr, dev, queue_data, url, img_data):
        if tr:
            self.detector.is_playing = tr["is_playing"]
            self.s_status.configure(text="spotify active", text_color=TXT_MAIN)
            self.s_name.configure(text=tr['name'])
            self.s_artist.configure(text=tr['artist'])

            if dev:
                self.s_device.configure(text=f"playing on: {dev['name']}")
            else:
                self.s_device.configure(text="no active device")

            self.pbar.set(tr["progress_ratio"])
            for i, lbl in enumerate(self.qlabels):
                if i < len(queue_data):
                    song = queue_data[i]
                    lbl.configure(text=f"{i+1}. {song['name']} — {song['artist']}")
                else:
                    lbl.configure(text="")

            if url and url != self.curr_img:
                if img_data == "error":
                    self.art.configure(image=None, text="art error")
                elif img_data:
                    d = Image.open(BytesIO(img_data))
                    p = ctk.CTkImage(light_image=d, size=(ART_SIZE, ART_SIZE))
                    self.art.configure(image=p, text="")
                    self.curr_img = url
        else:
            self.detector.is_playing = None
            self.s_status.configure(text="spotify inactive", text_color=TXT_MUTE)
            self.s_name.configure(text="no active playback")
            self.s_artist.configure(text="waiting for spotify connection")
            self.s_device.configure(text="no active device")
            self.pbar.set(0)
            self.art.configure(image=None, text="no art")
            self.curr_img = None


    def change_cd(self, val):
        self.detector.upd_cooldown(val)
        self.cd_txt.configure(text=f"{val:.1f}s")

    def toggle_mode(self, val):
        if val == "ml":
            ok = self.detector.switch_mode(True)
            if not ok:
                self.m_var.set("heuristic")
                self.ml_warn.configure(text="model file not found")
        else:
            self.detector.switch_mode(False)
            self.ml_warn.configure(text="")

    def close_app(self):
        self.perf_log.close()
        self.cap.release()
        self.destroy()


if __name__ == "__main__":
    a = VisionPlayApp()
    a.mainloop()

import os
import pyautogui
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import requests

load_dotenv()

class MediaController:
    def play(self):
        pyautogui.press("playpause")
    def pause(self):
        pyautogui.press("playpause")
    def next_track(self):
        pyautogui.press("nexttrack")
    def prev_track(self):
        pyautogui.press("prevtrack")
    def volume_up(self, steps=5):
        pyautogui.press("volumeup", presses=steps)
    def volume_down(self, steps=5):
        pyautogui.press("volumedown", presses=steps)

class SpotifyController:
    def __init__(self):
        self.sp = None
        cid = os.getenv("SPOTIFY_CLIENT_ID")
        csec = os.getenv("SPOTIFY_CLIENT_SECRET")
        redirect = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8080/callback")

        if not cid or not csec:
            print("spotify credentials not found. add them to your .env file.")
            return

        try:
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=cid,
                client_secret=csec,
                redirect_uri=redirect,
                scope="user-read-playback-state,user-modify-playback-state"
            ))
            print("spotify initialised")
        except (spotipy.SpotifyException, spotipy.oauth2.SpotifyOauthError, requests.RequestException) as e:
            print(f"spotify auth error: {e}")

    def get_song(self):
        if not self.sp:
            return None
        try:
            res = self.sp.current_playback()
            if res and res.get('item'):
                t = res['item']
                return {
                    "name": t['name'],
                    "artist": t['artists'][0]['name'],
                    "is_playing": res['is_playing'],
                    "progress_ratio": res['progress_ms'] / t['duration_ms'] if t['duration_ms'] > 0 else 0.0,
                    "image_url": t['album']['images'][0]['url'] if t['album']['images'] else None
                }
            else:
                return None  # nothing playing
        except (spotipy.SpotifyException, requests.RequestException) as e:
            print(f"spotify api error: {e}")
        return None
    def get_queue(self):
        if not self.sp:
            return []
        try:
            q = self.sp.queue()
            if q and "queue" in q:
                upc = []
                for item in q["queue"][:4]:
                    upc.append({
                        "name": item["name"],
                        "artist": item["artists"][0]["name"],
                        "image_url": item["album"]["images"][0]["url"] if item["album"]["images"] else None
                    })
                return upc  # outside loop collects all 4 first
            return []
        except (spotipy.SpotifyException, requests.RequestException) as e:
            print(f"spotify queue error: {e}")
            return []
    def get_dev(self):
        if not self.sp:
            return []
        try:
            res = self.sp.devices()
            if res and res.get('devices'):
                dev = res['devices'][0]
                return {
                    "id": dev["id"],
                    "name": dev["name"],
                    "is_active": dev["is_active"],
                    "type": dev["type"]
                }
            return None
        except (spotipy.SpotifyException, requests.RequestException) as e:
            print(f"spotify devices error: {e}")
            return None
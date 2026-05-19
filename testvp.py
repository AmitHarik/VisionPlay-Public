import time
import pytest
import spotipy
import numpy as np
from unittest.mock import MagicMock, patch
from gesturerecog import GestureProc
from main import MediaController, SpotifyController


class _LM:
    def __init__(self, x=0.5, y=0.5, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def _lms(overrides=None):
    # 21 lms all default to 0.5 0.5 for neutral fist
    lms = [_LM() for _ in range(21)]
    for idx, (x, y) in (overrides or {}).items():
        lms[idx].x, lms[idx].y = x, y
    return lms


@pytest.fixture
def proc():
    g = GestureProc()
    g.use_ml = False
    g.model = None
    return g


@pytest.fixture
def ctrl():
    return MagicMock()


# _get_gest heuristic

def test_fist(proc):
    # tips at same y as mcps so no fingers up and thumb not out
    assert proc._get_gest(_lms()) == "fist"

def test_op(proc):
    lms = _lms({
        8: (0.5, 0.2), 5: (0.5, 0.5),
        12: (0.5, 0.2), 9: (0.5, 0.5),
        16: (0.5, 0.2), 13: (0.5, 0.5),
        20: (0.5, 0.2), 17: (0.5, 0.5),
    })
    assert proc._get_gest(lms) == "open_palm"


def test_iu(proc):
    lms = _lms({8: (0.5, 0.2), 5: (0.5, 0.5)})
    assert proc._get_gest(lms) == "index_up"


def test_tu(proc):
    lms = _lms({
        8: (0.5, 0.2), 5: (0.5, 0.5),
        12: (0.5, 0.2), 9: (0.5, 0.5),
    })
    assert proc._get_gest(lms) == "two_up"


def test_tr(proc):
    # dx=0.2 dy=0 ext=0.2 thresholds pass dx > 0
    lms = _lms({4: (0.7, 0.5), 2: (0.5, 0.5), 5: (0.5, 0.5)})
    assert proc._get_gest(lms) == "thumb_right"


def test_tl(proc):
    lms = _lms({4: (0.3, 0.5), 2: (0.5, 0.5), 5: (0.5, 0.5)})
    assert proc._get_gest(lms) == "thumb_left"


def test_none(proc):
    # ring only so no matching gest
    lms = _lms({16: (0.5, 0.2), 13: (0.5, 0.5)})
    assert proc._get_gest(lms) is None


# fire()

def test_fire_same(proc, ctrl):
    proc.controller = ctrl
    proc.last_gest = "fist"
    proc.fire("fist")
    ctrl.play.assert_not_called()


def test_fire_cd(proc, ctrl):
    proc.controller = ctrl
    proc.last_gest = ""
    proc.last_t = time.time()
    proc.cooldown = 999
    proc.fire("fist")
    ctrl.play.assert_not_called()


def test_fire_act(proc, ctrl):
    proc.controller = ctrl
    proc.last_gest = ""
    proc.last_t = 0.0
    proc.is_playing = None
    proc.fire("fist")
    ctrl.play.assert_called_once()


def test_fire_skipplay(proc, ctrl):
    proc.controller = ctrl
    proc.last_gest = ""
    proc.last_t = 0.0
    proc.is_playing = True
    proc.fire("fist")
    ctrl.play.assert_not_called()


def test_fire_skippause(proc, ctrl):
    proc.controller = ctrl
    proc.last_gest = ""
    proc.last_t = 0.0
    proc.is_playing = False
    proc.fire("open_palm")
    ctrl.pause.assert_not_called()


def test_fire_state(proc, ctrl):
    proc.controller = ctrl
    proc.last_gest = ""
    proc.last_t = 0.0
    proc.is_playing = None
    proc.fire("index_up")
    assert proc.last_gest == "index_up"
    assert proc.last_t > 0


# switch_mode

def test_sw_ml_nomod(proc):
    proc.model = None
    assert proc.switch_mode(True) is False
    assert proc.use_ml is False


def test_sw_ml_mod(proc):
    proc.model = MagicMock()
    assert proc.switch_mode(True) is True
    assert proc.use_ml is True


def test_ml_zerodiv(proc):
    proc.use_ml = True
    proc.model = MagicMock()
    # synthetic lms to check zero div
    lms = _lms({i: (0.0, 0.0) for i in range(21)})
    try:
        proc.proc_lms(lms)
    except ZeroDivisionError:
        pytest.fail("zerodivisionerror raised")


def test_gb_noise(proc):
    proc.fire = MagicMock()
    proc._get_gest = MagicMock(side_effect=["fist", "fist", "open_palm"])
    
    res = MagicMock()
    res.multi_hand_landmarks = [MagicMock()]
    proc.hands.process = MagicMock(return_value=res)
    
    frm = np.zeros((10, 10, 3), dtype=np.uint8)
    
    proc.proc_frame(frm)
    proc.proc_frame(frm)
    proc.proc_frame(frm)
    
    # aborted trigger
    proc.fire.assert_not_called()


# mediacontroller

def test_mc_play():
    with patch('main.pyautogui') as pg:
        MediaController().play()
        pg.press.assert_called_once_with("playpause")


def test_mc_nt():
    with patch('main.pyautogui') as pg:
        MediaController().next_track()
        pg.press.assert_called_once_with("nexttrack")


def test_mc_pt():
    with patch('main.pyautogui') as pg:
        MediaController().prev_track()
        pg.press.assert_called_once_with("prevtrack")


def test_mc_vudef():
    with patch('main.pyautogui') as pg:
        MediaController().volume_up()
        pg.press.assert_called_once_with("volumeup", presses=5)


def test_sp_netfail():
    with patch('main.spotipy.Spotify') as m_sp:
        inst = m_sp.return_value
        inst.current_user_playing_track.side_effect = spotipy.SpotifyException(500, -1, "err")
        
        sp = SpotifyController()
        sp.sp = inst 
        
        assert sp.get_song() is None


def test_mc_vd():
    with patch('main.pyautogui') as pg:
        MediaController().volume_down()
        pg.press.assert_called_once_with("volumedown", presses=5)

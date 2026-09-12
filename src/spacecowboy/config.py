import math

# Target display — SSD1306 128x64 monochrome OLED.
SCREEN_W = 128
SCREEN_H = 64

FOV = math.radians(66)
MAX_RENDER_DIST = 18.0

MOVE_SPEED = 0.55          # cells / second
TURN_SPEED = math.radians(22)
DRIFT_AMOUNT = math.radians(8)  # peak heading wobble rate while walking, radians/sec

FPS = 20

# --- walker behavior timings (seconds), min/max ---
WALK_DURATION = (5.0, 13.0)
STOP_DURATION = (2.5, 6.0)
TURN_DURATION = (1.2, 3.0)

# --- world generation ---
OBSTACLE_TYPES = {
    2: {"name": "dead_tree", "height": 1.65},
}

# day/night cycle: full cycle length in seconds of real time
DAY_LENGTH = 240.0

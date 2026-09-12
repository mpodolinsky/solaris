"""Real hardware backend: SSD1306 128x64 I2C OLED via luma.oled.

Only imported when --backend oled is requested, so the simulator path
never needs luma/RPi libraries installed.
"""
import numpy as np
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306
from PIL import Image

from .. import config
from ..raycaster import ordered_dither
from .base import Display


class OledDisplay(Display):
    def __init__(self, i2c_port: int = 1, i2c_address: int = 0x3C):
        serial = i2c(port=i2c_port, address=i2c_address)
        self.device = ssd1306(serial, width=config.SCREEN_W, height=config.SCREEN_H)

    def show(self, fb: np.ndarray) -> None:
        # The real display is strictly 1-bit, so the continuous brightness
        # framebuffer gets reduced with ordered dithering here.
        bits = ordered_dither(fb)
        img = Image.fromarray((bits * 255).astype(np.uint8), mode="L").convert("1")
        self.device.display(img)

    def close(self) -> None:
        self.device.clear()

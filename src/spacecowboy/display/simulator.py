"""Desktop stand-in for the SSD1306 OLED: a scaled-up phosphor-green window."""
import numpy as np
import pygame

from .. import config
from .base import Display

# Amber colormap: brightness in [0,1] -> an RGB stop along a warm phosphor
# gradient, from near-black through dark rust-amber to a bright amber-white
# highlight. This is a simulator-only luxury — the real 1-bit OLED can't do
# shades, so it dithers instead (see display/oled.py).
_STOPS = np.array(
    [
        [8, 5, 2],
        [70, 34, 6],
        [168, 92, 12],
        [255, 176, 40],
        [255, 226, 150],
    ],
    dtype=np.float32,
)
_STOP_POS = np.linspace(0.0, 1.0, len(_STOPS))


def _amber(brightness: np.ndarray) -> np.ndarray:
    flat = np.clip(brightness, 0, 1).ravel()
    r = np.interp(flat, _STOP_POS, _STOPS[:, 0])
    g = np.interp(flat, _STOP_POS, _STOPS[:, 1])
    b = np.interp(flat, _STOP_POS, _STOPS[:, 2])
    return np.stack([r, g, b], axis=-1).reshape(*brightness.shape, 3).astype(np.uint8)


class SimulatorDisplay(Display):
    def __init__(self, scale: int = 1):
        pygame.init()
        pygame.display.set_caption("space_cowboy — OLED simulator")
        self.scale = scale
        self.surface = pygame.display.set_mode(
            (config.SCREEN_W * scale, config.SCREEN_H * scale)
        )
        self.pixel_surface = pygame.Surface((config.SCREEN_W, config.SCREEN_H))
        self._quit = False

    def show(self, fb: np.ndarray) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._quit = True

        rgb = _amber(fb)
        # pygame surfarray expects (W, H, 3)
        pygame.surfarray.blit_array(self.pixel_surface, rgb.transpose(1, 0, 2))
        scaled = pygame.transform.scale(
            self.pixel_surface,
            (config.SCREEN_W * self.scale, config.SCREEN_H * self.scale),
        )
        self.surface.blit(scaled, (0, 0))
        pygame.display.flip()

    def should_quit(self) -> bool:
        return self._quit

    def close(self) -> None:
        pygame.quit()

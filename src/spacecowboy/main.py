import argparse
import random
import time

import numpy as np

from . import config, raycaster
from .walker import Walker
from .world import World


def build_display(backend: str, scale: int = 1):
    if backend == "sim":
        from .display.simulator import SimulatorDisplay
        return SimulatorDisplay(scale=scale)
    elif backend == "oled":
        from .display.oled import OledDisplay
        return OledDisplay()
    raise ValueError(f"unknown backend: {backend}")


def main():
    parser = argparse.ArgumentParser(description="space_cowboy — an empty world, walking forever")
    parser.add_argument("--backend", choices=["sim", "oled"], default="sim")
    parser.add_argument("--seed", type=int, default=None, help="world seed (default: random each run)")
    parser.add_argument("--fps", type=int, default=config.FPS)
    parser.add_argument(
        "--scale", type=int, default=1,
        help="sim backend only: window pixel scale. 1 = exact OLED size (128x64 window), matching the real hardware.",
    )
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    print(f"world seed: {seed}")

    rng = random.Random(seed)
    world = World(seed=seed)
    walker = Walker(world, rng)
    display = build_display(args.backend, scale=args.scale)

    fb = np.zeros((config.SCREEN_H, config.SCREEN_W), dtype=np.float32)
    dt = 1.0 / args.fps
    world_time = 0.0

    try:
        while not display.should_quit():
            frame_start = time.monotonic()

            walker.update(dt)
            world_time += dt
            raycaster.render(fb, walker, world, world_time)
            display.show(fb)

            elapsed = time.monotonic() - frame_start
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        display.close()


if __name__ == "__main__":
    main()

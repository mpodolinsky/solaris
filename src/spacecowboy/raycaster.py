"""Pseudo-3D raycaster, rendering into a float32 (H, W) brightness framebuffer.

Classic grid-DDA raycasting (Lodev-style) for solid obstacles, plus cheap
floor-casting for ground texture and a per-column horizon offset to fake
rolling terrain. The framebuffer holds continuous brightness in [0, 1] —
the simulator renders it as real shades of amber, while the real 1-bit OLED
backend reduces it to on/off with ordered (Bayer) dithering at output time.
"""
import math

import numpy as np

from . import config
from .noise import fbm, value_noise, value_noise_vec
from .walker import State
from .world import World

BAYER4 = np.array(
    [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ],
    dtype=np.float32,
) / 16.0


def ordered_dither(brightness: np.ndarray) -> np.ndarray:
    """Reduce a float brightness image to 1-bit using 4x4 Bayer dithering."""
    h, w = brightness.shape
    tiled = np.tile(BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w]
    return brightness > tiled


_HORIZON_ROW = config.SCREEN_H // 2
_HILL_LOOKAHEAD = 9.0
_HILL_AMPLITUDE = 9.0
_TERRAIN_SEED_OFFSET = 555

# Floor-casting: eye height above the ground plane, in cells. Combined with
# the focal length (= SCREEN_H, matching the wall projection), this is what
# makes near-ground texture sweep past faster than far ground — real motion
# parallax instead of a flat, screen-locked texture.
_CAMERA_HEIGHT = 0.35
_GROUND_NOISE_SCALE = 0.9

# Footstep camera bob: subtle vertical/horizontal sway timed to distance walked.
_BOB_FREQ = 0.9
_BOB_AMP_V = 1.0
_BOB_AMP_H = 0.4
_IDLE_SWAY_AMP = 0.4

# Static fine-grain speckle overlay for ground texture — a scrubby/dusty
# grain independent of world position, cheap enough for a Pi Zero.
_GRAIN = np.random.RandomState(1234).uniform(0.7, 1.3, size=(config.SCREEN_H, config.SCREEN_W)).astype(np.float32)

# Whole-screen film-grain overlay, applied last as a post-process — a subtle
# CRT/terminal-static texture over the finished frame, re-rolled every frame
# (unlike _GRAIN above, which is a fixed pattern).
_SCREEN_GRAIN_AMOUNT = 0.05
_grain_rng = np.random.RandomState(2024)


def render(fb: np.ndarray, walker, world: World, world_time: float):
    """Fills fb (float32 array, shape (SCREEN_H, SCREEN_W)) in place, values in [0, 1]."""
    fb[:] = 0.0

    sun_angle = (world_time / config.DAY_LENGTH) * 2 * math.pi
    sun_altitude = math.sin(sun_angle)
    is_night = sun_altitude < 0.05

    horizon_rows = _terrain_profile(walker, world)
    _draw_sky(fb, walker.angle, horizon_rows, is_night)
    _draw_ground(fb, walker, world, horizon_rows, is_night)
    _draw_sun_or_moon(fb, walker.angle, sun_angle, sun_altitude, horizon_rows)
    _draw_walls(fb, walker, world, horizon_rows, is_night)
    _apply_bob(fb, walker, world_time)
    _apply_grain(fb)


def _apply_bob(fb: np.ndarray, walker, world_time: float):
    """A light footstep sway — shifts the whole frame a pixel or so, the way
    a handheld POV bobs with each step. Settles to a slower idle sway when
    the walker isn't actively moving."""
    if walker.state == State.WALK:
        phase = walker.distance * _BOB_FREQ * 2 * math.pi
        dy = math.sin(phase * 2) * _BOB_AMP_V
        dx = math.sin(phase) * _BOB_AMP_H
    else:
        dy = math.sin(world_time * 0.6) * _IDLE_SWAY_AMP
        dx = math.sin(world_time * 0.37) * _IDLE_SWAY_AMP * 0.5

    shift_y, shift_x = round(dy), round(dx)
    if shift_y or shift_x:
        fb[:] = _shift_with_edge_fill(fb, shift_y, shift_x)


def _apply_grain(fb: np.ndarray) -> None:
    """Subtle whole-screen film-grain/static, re-rolled every frame."""
    noise = _grain_rng.uniform(-_SCREEN_GRAIN_AMOUNT, _SCREEN_GRAIN_AMOUNT, size=fb.shape).astype(np.float32)
    np.clip(fb + noise, 0.0, 1.0, out=fb)


def _shift_with_edge_fill(fb: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift the frame like np.roll, but replicate the edge row/column into
    the space it vacates instead of wrapping — a plain roll would smear the
    bottom of the ground into the top of the sky (and vice versa)."""
    out = fb
    if dy:
        out = np.roll(out, dy, axis=0)
        if dy > 0:
            out[:dy, :] = fb[0:1, :]
        else:
            out[dy:, :] = fb[-1:, :]
    if dx:
        base = out
        out = np.roll(out, dx, axis=1)
        if dx > 0:
            out[:, :dx] = base[:, 0:1]
        else:
            out[:, dx:] = base[:, -1:]
    return out


def _terrain_profile(walker, world: World) -> np.ndarray:
    """Per-column horizon row, offset by low-frequency terrain noise -> rolling hills."""
    w = config.SCREEN_W
    cols = np.arange(w)
    ray_angles = walker.angle + (cols / w * 2 - 1) * (config.FOV / 2)

    far = np.empty(w, dtype=np.float32)
    for x, ang in enumerate(ray_angles):
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        far[x] = fbm(
            (walker.x + cos_a * _HILL_LOOKAHEAD) * 0.05,
            (walker.y + sin_a * _HILL_LOOKAHEAD) * 0.05,
            world.seed + _TERRAIN_SEED_OFFSET,
            octaves=2,
        )

    offsets = (far - 0.5) * 2 * _HILL_AMPLITUDE
    rows = np.clip(_HORIZON_ROW + offsets, 4, config.SCREEN_H - 4).astype(np.int32)
    return rows


def _draw_sky(fb: np.ndarray, view_angle: float, horizon_rows: np.ndarray, is_night: bool):
    w, h = config.SCREEN_W, config.SCREEN_H
    cols = np.arange(w)
    ray_angles = view_angle + (cols / w * 2 - 1) * (config.FOV / 2)

    sky_base = 0.06 if is_night else 0.11
    cloud_strength = 0.10 if is_night else 0.17

    for x, ang in enumerate(ray_angles):
        bot = int(horizon_rows[x])
        if bot <= 0:
            continue
        fb[0:bot, x] = sky_base

        # Cloud layer: a skybox-style pattern keyed on view azimuth only, so
        # it pans smoothly as the walker turns rather than jittering per-frame.
        band = value_noise(ang * 2.6, 3.0, 4242)
        if band > 0.5:
            center_frac = 0.25 + 0.4 * value_noise(ang * 2.6, 9.0, 4243)
            thickness = bot * (0.10 + 0.16 * band)
            center = bot * center_frac
            rows = np.arange(0, bot)
            dist = np.abs(rows - center)
            cloud = np.clip(1.0 - dist / max(thickness, 1.0), 0.0, 1.0) * cloud_strength * (band - 0.5) * 2
            fb[0:bot, x] = np.maximum(fb[0:bot, x], sky_base + cloud)

    if is_night:
        # Bucket resolution must be finer than one screen column's angular
        # width, or the same star gets hit by several adjacent columns and
        # smears into a horizontal streak instead of a single point.
        for x, ang in enumerate(ray_angles):
            bucket = round(ang * 150)
            for row in range(1, int(horizon_rows[x]) - 1):
                if value_noise(bucket, row, 7777) > 0.965:
                    fb[row, x] = 1.0

    _draw_mountains(fb, ray_angles, horizon_rows, sky_base, is_night)

    # Horizon line: a soft glow rather than a hard bright stroke, with a
    # little grain so it doesn't read as a ruler-straight seam. The base
    # brightness and the grain's swing both need to stay clear of the amber
    # colormap's dark zone (its low end covers a wide range down to
    # near-black) — _GRAIN's full 0.7-1.3 swing was pushing about half the
    # columns' line pixels below that threshold, giving a speckled
    # "sometimes bright, sometimes black" horizon instead of a clean glow.
    line_bright = 0.42 if not is_night else 0.24
    line_grain = 0.9 + 0.2 * (_GRAIN[0, cols] - 0.7) / 0.6  # remapped to a narrower 0.9-1.1
    line_value = line_bright * line_grain
    fb[horizon_rows, cols] = np.maximum(fb[horizon_rows, cols], line_value)


_MOUNTAIN_MAX_PX = 13


def _draw_mountains(fb: np.ndarray, ray_angles: np.ndarray, horizon_rows: np.ndarray, sky_base: float, is_night: bool):
    """A distant mountain range silhouette, keyed only on view azimuth (like
    the clouds) so it pans as you turn but doesn't shift with position —
    exactly how a genuinely far-away backdrop should behave. Sits just
    behind the near rolling-hill horizon, giving the skyline some depth."""
    for x, ang in enumerate(ray_angles):
        bot = int(horizon_rows[x])
        if bot <= 1:
            continue
        # Base frequency picks out a handful of big peaks across the field
        # of view; the detail layer roughens their edges. Both need to
        # oscillate several times over the FOV or the whole range just
        # reads as one smooth, barely-visible band instead of a skyline.
        ridge_base = value_noise(ang * 9.0, 0.0, 9111)
        ridge_detail = value_noise(ang * 28.0, 5.0, 9222)
        ridge = ridge_base * 0.75 + ridge_detail * 0.25
        peak_px = int((ridge ** 1.3) * _MOUNTAIN_MAX_PX)
        if peak_px <= 0:
            continue
        top = max(0, bot - peak_px)
        mountain_bright = sky_base + (0.035 + 0.035 * ridge if not is_night else 0.018 + 0.02 * ridge)
        fb[top:bot, x] = mountain_bright


def _draw_ground(fb: np.ndarray, walker, world: World, horizon_rows: np.ndarray, is_night: bool):
    """True floor-casting: every ground row maps to a real distance on the
    ground plane, and the texture is sampled at the world coordinate that
    distance away. That's what makes it sweep past correctly as you walk —
    near rows move fast, far rows near the horizon barely move at all."""
    w, h = config.SCREEN_W, config.SCREEN_H
    # Start from the highest point any column's (hill-shifted) horizon
    # reaches, not the fixed global horizon — otherwise columns where the
    # hill pushes the horizon up leave an undrawn gap between sky and ground.
    start_row = max(1, int(horizon_rows.min()) + 1)
    if start_row >= h:
        return

    cols = np.arange(w)
    ray_angles = walker.angle + (cols / w * 2 - 1) * (config.FOV / 2)
    cos_a = np.cos(ray_angles).astype(np.float32)
    sin_a = np.sin(ray_angles).astype(np.float32)

    rows_arr = np.arange(start_row, h, dtype=np.float32)
    row_offset = np.clip(rows_arr - _HORIZON_ROW, 0.5, None)
    row_dist = np.clip((_CAMERA_HEIGHT * h) / row_offset, 0.05, config.MAX_RENDER_DIST * 2)

    wx = walker.x + row_dist[:, None] * cos_a[None, :]
    wy = walker.y + row_dist[:, None] * sin_a[None, :]

    clump = value_noise_vec(wx * _GROUND_NOISE_SCALE, wy * _GROUND_NOISE_SCALE, world.seed + 8181)
    terrain = value_noise_vec(wx * 0.12, wy * 0.12, world.seed + _TERRAIN_SEED_OFFSET)

    # A smooth asymptotic falloff (never hard-clips to exactly 1) instead of
    # clip(row_dist / MAX_RENDER_DIST, 0, 1) — that hard clip saturated
    # within the very first ground row (its distance already exceeded
    # MAX_RENDER_DIST there), while the next row dropped well under it,
    # producing a discontinuous jump: dim-then-bright immediately below the
    # horizon glow, reading as a dark seam.
    depth_norm = (row_dist / (row_dist + config.MAX_RENDER_DIST))[:, None]
    base = 0.30 + (1.0 - depth_norm) * 0.30
    patch = np.clip(clump - 0.4, 0, None) * 0.35
    dune = (terrain - 0.5) * 0.22
    light_mul = 0.55 if is_night else 0.8
    # Floor above 0 so the dune term can't swing a patch of ground dark
    # enough, right next to the horizon glow, to read as a black seam.
    brightness = np.clip(base + patch + dune, 0.24, 0.7).astype(np.float32) * light_mul

    valid = rows_arr[:, None] > horizon_rows[None, :]
    region = fb[start_row:h, :]
    fb[start_row:h, :] = np.where(valid, brightness, region)


def _draw_sun_or_moon(fb: np.ndarray, view_angle: float, sun_angle: float, altitude: float, horizon_rows: np.ndarray):
    w, h = config.SCREEN_W, config.SCREEN_H
    body_world_angle = sun_angle if altitude >= 0 else sun_angle + math.pi
    rel = (body_world_angle - view_angle + math.pi) % (2 * math.pi) - math.pi
    if abs(rel) > config.FOV / 2 + 0.15:
        return

    # cx can legitimately fall outside [0, w) here — the sun is allowed a bit
    # past the FOV edge (see the +0.15 margin above) so it slides smoothly
    # off-screen rather than snapping away. Only the horizon-row lookup needs
    # clamping (it indexes a fixed-size array); the draw position must stay
    # unclamped, or the sun's center would stick to the edge column instead
    # of continuing to move as the view keeps turning.
    cx = int((rel / (config.FOV / 2) * 0.5 + 0.5) * w)
    horizon_idx = max(0, min(w - 1, cx))
    local_horizon = horizon_rows[horizon_idx]
    cy = int(local_horizon - abs(altitude) * (local_horizon - 6) - 6)
    radius = 4
    glow = 1.0 if altitude >= 0 else 0.8
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                x, y = cx + dx, cy + dy
                if 0 <= x < w and 0 <= y < config.SCREEN_H:
                    fb[y, x] = glow


def _dead_tree_profile(map_x: int, map_y: int, wall_x: float, seed: int):
    """A bare, blasted tree: a trunk that opens into a wide, mushroom-like
    crown of overlapping limbs spreading horizontally, not a candelabra of
    vertical spikes — real dead trees are wider than they are tall up top.
    A few thin twigs poke through the crown's top edge for raggedness. Each
    tree cell gets its own layout, hashed from its coordinates, so no two
    trees look quite the same."""
    trunk_half_width = 0.09
    dist_from_center = abs(wall_x - 0.5)
    in_trunk = dist_from_center <= trunk_half_width
    # A little root flare at the base, tapering to nothing a short way out.
    bottom_scale = 1.0 if in_trunk else max(0.0, 0.2 - dist_from_center)

    top_scale = 0.18 if in_trunk else 0.0  # trunk stub before the crown starts

    # Main limbs: wide, low, overlapping humps positioned to tile across the
    # whole crown so it reads as one continuous horizontal spread rather than
    # isolated blobs.
    for k, base_cx in enumerate((0.18, 0.5, 0.82)):
        cx = base_cx + 0.14 * (value_noise(map_x * 2.1 + k * 9.7, map_y * 2.1 + k * 4.3, seed + 3131) - 0.5)
        limb_h = 0.42 + 0.30 * value_noise(map_x * 2.9 + k * 6.1, map_y * 2.9 + k * 8.9, seed + 4141)
        limb_w = 0.24 + 0.14 * value_noise(map_x * 3.3 + k * 5.5, map_y * 3.3 + k * 2.7, seed + 5252)
        d = (wall_x - cx) / limb_w
        limb_shape = max(0.0, 1.0 - d * d)  # rounded, flat-ish plateau, not a peak
        top_scale = max(top_scale, limb_h * limb_shape)

    # A handful of thin twigs breaking through the crown's silhouette.
    for k in range(4):
        cx = 0.06 + 0.88 * value_noise(map_x * 1.9 + k * 11.3, map_y * 1.9 + k * 5.1, seed + 5151)
        branch_h = top_scale + 0.15 + 0.25 * value_noise(map_x * 2.7 + k * 7.7, map_y * 2.7 + k * 3.3, seed + 6161)
        branch_w = 0.03 + 0.03 * value_noise(map_x * 3.5 + k * 4.1, map_y * 3.5 + k * 9.9, seed + 7171)
        spike = max(0.0, 1.0 - abs((wall_x - cx) / branch_w)) ** 1.8
        top_scale = max(top_scale, branch_h * spike)

    return top_scale, bottom_scale


def _silhouette_profile(cell: int, map_x: int, map_y: int, wall_x: float, seed: int):
    """Returns (top_scale, bottom_scale) in [0,1] applied to the half-height
    above/below the horizon. Dead trees are the only obstacle type now."""
    return _dead_tree_profile(map_x, map_y, wall_x, seed)


def _draw_walls(fb: np.ndarray, walker, world: World, horizon_rows: np.ndarray, is_night: bool):
    w, h = config.SCREEN_W, config.SCREEN_H
    px, py, view_angle = walker.x, walker.y, walker.angle

    for x in range(w):
        camera_x = (x / w) * 2 - 1
        ray_angle = view_angle + camera_x * (config.FOV / 2)
        rdx, rdy = math.cos(ray_angle), math.sin(ray_angle)

        map_x, map_y = int(math.floor(px)), int(math.floor(py))

        delta_x = abs(1 / rdx) if rdx != 0 else 1e30
        delta_y = abs(1 / rdy) if rdy != 0 else 1e30

        if rdx < 0:
            step_x = -1
            side_x = (px - map_x) * delta_x
        else:
            step_x = 1
            side_x = (map_x + 1 - px) * delta_x

        if rdy < 0:
            step_y = -1
            side_y = (py - map_y) * delta_y
        else:
            step_y = 1
            side_y = (map_y + 1 - py) * delta_y

        cell = 0
        side = 0
        for _ in range(int(config.MAX_RENDER_DIST) + 4):
            if side_x < side_y:
                side_x += delta_x
                map_x += step_x
                side = 0
            else:
                side_y += delta_y
                map_y += step_y
                side = 1
            cell = world.get_cell(map_x, map_y)
            if cell != 0:
                break
            if min(side_x, side_y) > config.MAX_RENDER_DIST:
                break

        if cell == 0:
            continue

        if side == 0:
            perp_dist = (map_x - px + (1 - step_x) / 2) / rdx
            wall_x = py + perp_dist * rdy
        else:
            perp_dist = (map_y - py + (1 - step_y) / 2) / rdy
            wall_x = px + perp_dist * rdx
        perp_dist = max(perp_dist, 1e-4)
        wall_x -= math.floor(wall_x)

        # Per-cell jitter softens the "uniform block" look into something rougher.
        jitter = 0.8 + 0.4 * value_noise(map_x * 0.37, map_y * 0.37, world.seed + 3131)
        height_factor = config.OBSTACLE_TYPES[cell]["height"] * jitter

        line_height = int((h / perp_dist) * height_factor)
        half = line_height // 2
        horizon_here = int(horizon_rows[x])

        top_scale, bottom_scale = _silhouette_profile(cell, map_x, map_y, wall_x, world.seed)
        draw_start = max(0, horizon_here - int(half * top_scale))
        draw_end = min(h - 1, horizon_here + int(half * bottom_scale))
        if draw_end < draw_start:
            continue

        brightness = max(0.0, 1.0 - perp_dist / config.MAX_RENDER_DIST)
        brightness = 0.15 + brightness * 0.85
        if side == 1:
            brightness *= 0.78  # cheap fake shading between wall orientations
        brightness *= 0.6 if is_night else 0.82

        fb[draw_start:draw_end + 1, x] = brightness

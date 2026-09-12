"""The autonomous wanderer. No input, no goal — it just keeps existing."""
import enum
import math
import random

from . import config
from .world import World


class State(enum.Enum):
    WALK = "walk"
    STOP = "stop"   # paused, slowly scanning left/right
    TURN = "turn"   # rotating toward a new heading


class Walker:
    def __init__(self, world: World, rng: random.Random):
        self.world = world
        self.rng = rng
        self.distance = 0.0  # cumulative distance walked, drives the footstep bob
        self.x = 0.5
        self.y = 0.5
        self.angle = rng.uniform(0, 2 * math.pi)

        self.state = State.WALK
        self._state_timer = 0.0
        self._target_angle = self.angle
        self._scan_dir = 1.0

        # A slow, smooth heading wobble applied continuously while walking,
        # so a straight leg drifts and curves organically instead of being
        # ruler-straight until a discrete turn snaps it elsewhere.
        self._drift_phase = rng.uniform(0, 2 * math.pi)
        self._drift_freq = rng.uniform(0.12, 0.28)

        self._enter_state(State.WALK)

    def _enter_state(self, state: State):
        self.state = state
        if state == State.WALK:
            self._state_timer = self.rng.uniform(*config.WALK_DURATION)
        elif state == State.STOP:
            self._state_timer = self.rng.uniform(*config.STOP_DURATION)
            self._scan_dir = self.rng.choice((-1.0, 1.0))
        elif state == State.TURN:
            self._state_timer = self.rng.uniform(*config.TURN_DURATION)
            # A course correction, not a pivot — the walker should still read
            # as "heading onward," just onto a new bearing.
            turn_amount = self.rng.uniform(math.radians(20), math.radians(65))
            turn_amount *= self.rng.choice((-1.0, 1.0))
            self._target_angle = self.angle + turn_amount

    def _cell_ahead_clear(self, distance: float) -> bool:
        nx = self.x + math.cos(self.angle) * distance
        ny = self.y + math.sin(self.angle) * distance
        return not self.world.is_solid(int(math.floor(nx)), int(math.floor(ny)))

    def update(self, dt: float):
        self._state_timer -= dt

        if self.state == State.WALK:
            if not self._cell_ahead_clear(0.8):
                self._enter_state(State.TURN)
                return
            step = config.MOVE_SPEED * dt
            self.angle += math.sin(self._drift_phase + self.distance * self._drift_freq) * config.DRIFT_AMOUNT * dt
            self.x += math.cos(self.angle) * step
            self.y += math.sin(self.angle) * step
            self.distance += step
            if self._state_timer <= 0:
                # Mostly keep going forward; occasionally pause to look
                # around; rarely peel off onto a new bearing.
                r = self.rng.random()
                if r < 0.55:
                    self._enter_state(State.WALK)
                elif r < 0.85:
                    self._enter_state(State.STOP)
                else:
                    self._enter_state(State.TURN)

        elif self.state == State.STOP:
            self.angle += self._scan_dir * math.radians(12) * dt
            if self._state_timer <= 0:
                self._enter_state(State.WALK)

        elif self.state == State.TURN:
            diff = (self._target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
            step = config.TURN_SPEED * dt
            if abs(diff) <= step:
                self.angle = self._target_angle
                self._enter_state(State.WALK)
            else:
                self.angle += step if diff > 0 else -step
            if self._state_timer <= 0:
                self._enter_state(State.WALK)

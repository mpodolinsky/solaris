# space_cowboy

A little sealed world that lives on a Raspberry Pi and a 128x64 monochrome
OLED. It boots, generates an infinite desert, and wanders through it forever
— no input, no goal, no UI. Aesthetic: Fallout terminal / Pip-Boy meets a
Wolfenstein-style raycaster.

## How it works

- `world.py` — the world is not stored anywhere. Every grid cell's contents
  (empty ground, rock, dead tree, wreck, telegraph pole, rare monolith) is a
  pure function of its `(x, y)` coordinate and a random seed, via layered
  value noise. It's infinite because nothing needs to be generated ahead of
  time — you can query any cell, anywhere, at any time and get a consistent
  answer.
- `walker.py` — an autonomous agent with a simple state machine
  (walk / stop-and-scan / turn), no player input at all.
- `raycaster.py` — a classic grid-DDA raycaster (Wolfenstein-style) rendering
  into a 1-bit framebuffer, with ordered (Bayer) dithering standing in for
  grayscale since the real display is strictly black/white. Includes a
  day/night cycle (sun/moon arc, stars at night).
- `display/` — pluggable output backend: `simulator.py` (a scaled-up pygame
  window, phosphor green) for developing on a desktop, `oled.py` (real
  SSD1306 over I2C via `luma.oled`) for the actual hardware.

## Running the simulator (desktop)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m spacecowboy.main --backend sim
```

Press `Esc` or close the window to quit. Pass `--seed 12345` to pin a
specific world, or leave it off for a fresh random one every run.

## Deploying to the Raspberry Pi

Hardware: any Pi (Zero / Zero 2 W is plenty) + an SSD1306 128x64 I2C OLED.
Wire it to the Pi's I2C pins (3.3V, GND, SDA → GPIO2, SCL → GPIO3).

1. Enable I2C: `sudo raspi-config` → Interface Options → I2C → enable.
   Confirm the OLED shows up: `sudo i2cdetect -y 1` (usually address `0x3C`).
2. On the Pi:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements-pi.txt
   PYTHONPATH=src .venv/bin/python -m spacecowboy.main --backend oled
   ```

### Autostart on boot (systemd)

Create `/etc/systemd/system/space-cowboy.service`:

```ini
[Unit]
Description=space_cowboy
After=multi-user.target

[Service]
WorkingDirectory=/home/pi/space_cowboy
Environment=PYTHONPATH=/home/pi/space_cowboy/src
ExecStart=/home/pi/space_cowboy/.venv/bin/python -m spacecowboy.main --backend oled
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl enable --now space-cowboy
```

Now it boots straight into the wasteland every time you power it on, no
terminal needed — just the box on the desk.

## Tuning the vibe

Everything worth tweaking lives in `config.py`: field of view, render
distance, how often the walker stops/turns, obstacle density/heights
(`world.py`'s thresholds), day length, etc.

"""
Real-time Vocal Pitch Visualizer

- Listens to microphone input
- Detects pitch with aubio
- Maps to musical notes and shows the active key on a piano keyboard
- Practice mode with target note, feedback (perfect/close/miss) and score
- Clickable piano keys: play synthesized note sounds

Controls:
    ← / → : change target note (semitone)
    ↑ / ↓ : change target note (octave)
    R     : reset practice stats
    Mouse : click a piano key to hear that note
"""

import math
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
import aubio
import pygame
import cv2


# -----------------------------
# Configuration
# -----------------------------

SAMPLE_RATE = 44100
BUFFER_SIZE = 2048
HOP_SIZE = 1024

START_MIDI = 48   # C3
END_MIDI = 84     # C6

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

CAM_WIDTH = 640
CAM_HEIGHT = 480

MARGIN = 24
GAP = 24
TITLE_BAR_HEIGHT = 90
WHITE_KEY_WIDTH = 30
WHITE_KEY_HEIGHT = 150

# 🌸 Powder pink theme
BG_COLOR = (252, 244, 248)
PANEL_COLOR = (255, 252, 255)
PANEL_BORDER = (232, 210, 224)
TEXT_COLOR = (55, 40, 65)
MUTED_TEXT = (150, 135, 155)
TITLE_PINK = (235, 150, 185)

# Accuracy colors
CORRECT_COLOR = (190, 235, 200)
CLOSE_COLOR = (255, 220, 190)
WRONG_COLOR = (255, 185, 195)

# Click flash
FLASH_FRAMES = 12  # ~200ms @ 60 FPS


# -----------------------------
# Shared state
# -----------------------------

class PitchState:
    def __init__(self):
        self.pitch_hz: float = 0.0
        self.midi: Optional[int] = None
        self.lock = threading.Lock()


pitch_state = PitchState()


class PracticeStats:
    """Stores practice mode state and scoring."""
    def __init__(self):
        self.target_midi: int = 60  # C4
        self.attempts: int = 0
        self.perfect: int = 0
        self.close: int = 0
        self.cooldown_frames: int = 0  # to avoid counting every frame
        self.last_result: str = ""     # "Perfect", "Close", "Miss"


practice = PracticeStats()

# midi -> remaining flash frames
clicked_key_flash: dict[int, int] = {}


# -----------------------------
# Pitch / note utilities
# -----------------------------

def hz_to_midi(freq_hz: float) -> Optional[int]:
    if freq_hz <= 0:
        return None
    return int(round(69 + 12 * math.log2(freq_hz / 440.0)))


def midi_to_note_name(midi: int) -> str:
    note_index = midi % 12
    octave = (midi // 12) - 1
    return f"{NOTE_NAMES[note_index]}{octave}"


def midi_to_frequency(midi: int) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12.0))


def create_pitch_detector():
    pitch = aubio.pitch("yin", BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE)
    pitch.set_unit("Hz")
    pitch.set_silence(-40)
    return pitch


pitch_detector = create_pitch_detector()


# -----------------------------
# Audio callback (microphone)
# -----------------------------

def audio_callback(indata, frames, time_info, status):
    mono = indata[:, 0].astype(np.float32)
    pitch = float(pitch_detector(mono)[0])
    conf = float(pitch_detector.get_confidence())

    with pitch_state.lock:
        if conf > 0.8 and pitch > 0:
            pitch_state.pitch_hz = pitch
            pitch_state.midi = hz_to_midi(pitch)
        else:
            pitch_state.pitch_hz = 0.0
            pitch_state.midi = None


# -----------------------------
# Tone playback for piano keys
# -----------------------------

def play_tone(freq: float, duration: float = 0.9, volume: float = 0.6) -> None:
    """Play a short, piano-like synthesized tone for the given frequency."""
    if freq <= 0:
        return

    def _worker():
        fs = SAMPLE_RATE
        t = np.linspace(0, duration, int(fs * duration), False)

        # Base sine + a few harmonics for a richer timbre
        fundamental = np.sin(2 * np.pi * freq * t)
        harmonic2 = 0.4 * np.sin(2 * np.pi * 2 * freq * t)
        harmonic3 = 0.2 * np.sin(2 * np.pi * 3 * freq * t)

        signal = fundamental + harmonic2 + harmonic3

        # Simple attack / release envelope
        attack_time = 0.02
        release_time = 0.25
        attack_samples = int(fs * attack_time)
        release_samples = int(fs * release_time)

        envelope = np.ones_like(signal)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(1.0, 0.0, release_samples)

        signal *= envelope

        # Normalize and apply volume
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val

        signal = (signal * volume).astype(np.float32)

        try:
            sd.stop()
        except Exception:
            pass

        sd.play(signal, fs, blocking=False)

    threading.Thread(target=_worker, daemon=True).start()


# -----------------------------
# UI helpers
# -----------------------------

def build_piano_keys(start: int, end: int, w: int, h: int):
    """Return a list of white + black keys with positions."""
    keys = []
    white_index = 0
    for midi in range(start, end + 1):
        name = midi_to_note_name(midi)
        black = "#" in name
        if not black:
            rect = pygame.Rect(white_index * w, 0, w, h)
            keys.append({"midi": midi, "rect": rect, "black": False})
            white_index += 1
        else:
            bw = int(w * 0.55)
            bh = int(h * 0.65)
            rect = pygame.Rect(white_index * w - bw // 2, 0, bw, bh)
            keys.append({"midi": midi, "rect": rect, "black": True})
    return keys


def draw_panel(screen, rect):
    pygame.draw.rect(screen, PANEL_COLOR, rect, border_radius=14)
    pygame.draw.rect(screen, PANEL_BORDER, rect, width=1, border_radius=14)


# -----------------------------
# Main Application
# -----------------------------

def run_app():
    pygame.init()
    pygame.display.set_caption("Real-time Vocal Pitch Visualizer")

    title_font = pygame.font.SysFont("Segoe UI", 28, bold=True)
    sub_font = pygame.font.SysFont("Segoe UI", 18)
    info_font = pygame.font.SysFont("Segoe UI", 20)
    small_font = pygame.font.SysFont("Segoe UI", 14)

    # Piano geometry
    white_count = sum(
        1 for m in range(START_MIDI, END_MIDI + 1)
        if "#" not in midi_to_note_name(m)
    )
    piano_width = white_count * WHITE_KEY_WIDTH

    PIANO_PADDING_X = 18
    piano_panel_width = piano_width + 2 * PIANO_PADDING_X
    piano_height = WHITE_KEY_HEIGHT + 160  # a bit more space for active note box

    total_width = MARGIN + CAM_WIDTH + GAP + piano_panel_width + MARGIN
    content_height = max(CAM_HEIGHT, piano_height)
    total_height = MARGIN + TITLE_BAR_HEIGHT + GAP + content_height + MARGIN

    screen = pygame.display.set_mode((total_width, total_height))

    # Layout rects
    title_rect = pygame.Rect(
        MARGIN, MARGIN,
        total_width - 2 * MARGIN,
        TITLE_BAR_HEIGHT,
    )

    cam_rect = pygame.Rect(
        MARGIN,
        MARGIN + TITLE_BAR_HEIGHT + GAP,
        CAM_WIDTH, CAM_HEIGHT,
    )

    piano_rect = pygame.Rect(
        cam_rect.right + GAP,
        cam_rect.top,
        piano_panel_width, content_height,
    )

    keys = build_piano_keys(START_MIDI, END_MIDI, WHITE_KEY_WIDTH, WHITE_KEY_HEIGHT)

    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    # Audio stream (mic)
    stream = sd.InputStream(
        channels=1,
        callback=audio_callback,
        samplerate=SAMPLE_RATE,
        blocksize=HOP_SIZE,
        dtype="float32",
    )
    stream.start()

    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # Practice mode controls
                if event.key == pygame.K_RIGHT:
                    practice.target_midi = min(END_MIDI, practice.target_midi + 1)
                    practice.cooldown_frames = 0
                elif event.key == pygame.K_LEFT:
                    practice.target_midi = max(START_MIDI, practice.target_midi - 1)
                    practice.cooldown_frames = 0
                elif event.key == pygame.K_UP:
                    practice.target_midi = min(END_MIDI, practice.target_midi + 12)
                    practice.cooldown_frames = 0
                elif event.key == pygame.K_DOWN:
                    practice.target_midi = max(START_MIDI, practice.target_midi - 12)
                    practice.cooldown_frames = 0
                elif event.key == pygame.K_r:
                    practice.attempts = 0
                    practice.perfect = 0
                    practice.close = 0
                    practice.last_result = ""
                    practice.cooldown_frames = 0

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Click on piano keys -> play sound + flash
                mx, my = event.pos
                if piano_rect.collidepoint(mx, my):
                    px = piano_rect.x + PIANO_PADDING_X
                    py = piano_rect.y + 40

                    clicked_midi = None

                    # First check black keys (on top visually)
                    for key in keys:
                        if key["black"]:
                            rect = key["rect"].move(px, py)
                            if rect.collidepoint(mx, my):
                                clicked_midi = key["midi"]
                                break

                    # Then check white keys
                    if clicked_midi is None:
                        for key in keys:
                            if not key["black"]:
                                rect = key["rect"].move(px, py)
                                if rect.collidepoint(mx, my):
                                    clicked_midi = key["midi"]
                                    break

                    if clicked_midi is not None:
                        freq = midi_to_frequency(clicked_midi)
                        play_tone(freq)
                        clicked_key_flash[clicked_midi] = FLASH_FRAMES

        screen.fill(BG_COLOR)

        # --- Title section (with spacing) ---
        title = title_font.render("Real-time Vocal Pitch Visualizer", True, TEXT_COLOR)
        subtitle = sub_font.render(
            "Sing a note and watch the keyboard react.", True, MUTED_TEXT
        )

        title_y = title_rect.y + 8
        subtitle_y = title_y + title.get_height() + 8
        underline_y = subtitle_y + subtitle.get_height() + 6

        screen.blit(title, (title_rect.x, title_y))
        screen.blit(subtitle, (title_rect.x, subtitle_y))

        pygame.draw.line(
            screen,
            TITLE_PINK,
            (title_rect.x, underline_y),
            (title_rect.x + title.get_width(), underline_y),
            4,
        )

        # --- Camera panel ---
        draw_panel(screen, cam_rect)

        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_surface = pygame.image.frombuffer(
                frame_rgb.tobytes(), (CAM_WIDTH, CAM_HEIGHT), "RGB"
            )
            screen.blit(cam_surface, cam_rect)

        cam_label = sub_font.render("Camera", True, MUTED_TEXT)
        screen.blit(
            cam_label,
            (cam_rect.x, cam_rect.y - cam_label.get_height() - 4),
        )

        # --- Pitch data ---
        with pitch_state.lock:
            hz = pitch_state.pitch_hz
            midi = pitch_state.midi

        key_color = CLOSE_COLOR
        msg = "Waiting for a stable pitch..."
        cents = 0.0
        has_pitch = False
        note = None

        if hz > 0 and midi is not None:
            has_pitch = True
            note = midi_to_note_name(midi)
            ideal = midi_to_frequency(midi)
            cents = 1200 * math.log2(hz / ideal)

            if abs(cents) <= 10:
                key_color = CORRECT_COLOR
            elif abs(cents) <= 30:
                key_color = CLOSE_COLOR
            else:
                key_color = WRONG_COLOR

            msg = f"Note: {note}   •   {hz:.2f} Hz   (Δ {cents:+.1f} cents)"

            # Practice scoring with cooldown (avoid counting every frame)
            if practice.cooldown_frames > 0:
                practice.cooldown_frames -= 1
            else:
                practice.attempts += 1
                if midi == practice.target_midi and abs(cents) <= 10:
                    practice.perfect += 1
                    practice.last_result = "Perfect"
                elif midi == practice.target_midi and abs(cents) <= 30:
                    practice.close += 1
                    practice.last_result = "Close"
                else:
                    practice.last_result = "Miss"
                practice.cooldown_frames = 30  # ~0.5 s at 60 FPS
        else:
            if practice.cooldown_frames > 0:
                practice.cooldown_frames -= 1

        # --- Piano panel ---
        draw_panel(screen, piano_rect)

        piano_label = sub_font.render("Piano & pitch", True, MUTED_TEXT)
        screen.blit(
            piano_label,
            (piano_rect.x, piano_rect.y - piano_label.get_height() - 4),
        )

        target_text = sub_font.render(
            f"Target: {midi_to_note_name(practice.target_midi)}", True, TEXT_COLOR
        )
        screen.blit(
            target_text,
            (piano_rect.x + PIANO_PADDING_X, piano_rect.y + 10),
        )

        # Score summary
        if practice.attempts > 0:
            accuracy = 100.0 * practice.perfect / practice.attempts
            score_str = f"{practice.perfect}/{practice.attempts} ({accuracy:.0f}%)"
        else:
            score_str = "–"

        score_text = small_font.render(f"Perfect: {score_str}", True, MUTED_TEXT)
        screen.blit(
            score_text,
            (
                piano_rect.x + piano_rect.width - score_text.get_width() - PIANO_PADDING_X,
                piano_rect.y + 12,
            ),
        )

        # Draw piano keys
        px = piano_rect.x + PIANO_PADDING_X
        py = piano_rect.y + 40

        # White keys with soft shadow + flash
        for key in keys:
            if not key["black"]:
                r = key["rect"].move(px, py)

                # soft shadow behind key
                shadow = r.move(2, 3)
                pygame.draw.rect(screen, (235, 225, 235), shadow, border_radius=5)

                base = (255, 255, 255)
                if midi == key["midi"] and has_pitch:
                    base = key_color

                # click flash overrides color
                if key["midi"] in clicked_key_flash and clicked_key_flash[key["midi"]] > 0:
                    clicked_key_flash[key["midi"]] -= 1
                    base = (255, 210, 230)

                # target outline
                if key["midi"] == practice.target_midi:
                    pygame.draw.rect(screen, PANEL_BORDER, r.inflate(6, 6), 2, border_radius=7)

                pygame.draw.rect(screen, base, r, border_radius=5)
                pygame.draw.rect(screen, PANEL_BORDER, r, 1, border_radius=5)

        # Black keys with soft shadow + flash
        for key in keys:
            if key["black"]:
                r = key["rect"].move(px, py)

                shadow = r.move(2, 3)
                pygame.draw.rect(screen, (210, 200, 220), shadow, border_radius=5)

                base = (70, 60, 90)
                if midi == key["midi"] and has_pitch:
                    base = key_color

                if key["midi"] in clicked_key_flash and clicked_key_flash[key["midi"]] > 0:
                    clicked_key_flash[key["midi"]] -= 1
                    base = (255, 210, 230)

                pygame.draw.rect(screen, base, r, border_radius=5)

        # ----- Active Note Panel -----
        active_panel_y = py + WHITE_KEY_HEIGHT + 10
        active_panel_rect = pygame.Rect(
            piano_rect.x + PIANO_PADDING_X,
            active_panel_y,
            piano_panel_width - 2 * PIANO_PADDING_X,
            40
        )

        pygame.draw.rect(screen, PANEL_COLOR, active_panel_rect, border_radius=8)
        pygame.draw.rect(screen, PANEL_BORDER, active_panel_rect, 1, border_radius=8)

        if has_pitch and note is not None:
            active_text = info_font.render(
                f"Active Note: {note}  ({hz:.1f} Hz)   Δ {cents:+.1f} cents",
                True,
                TEXT_COLOR,
            )
        else:
            active_text = info_font.render("Active Note: —", True, MUTED_TEXT)

        screen.blit(
            active_text,
            (active_panel_rect.x + 10, active_panel_rect.y + 8),
        )

        # --- Tuning bar under the active note panel ---
        bar_width = piano_panel_width - 2 * PIANO_PADDING_X
        bar_x = piano_rect.x + PIANO_PADDING_X
        bar_y = active_panel_rect.bottom + 16
        bar_height = 6

        pygame.draw.rect(
            screen,
            PANEL_BORDER,
            pygame.Rect(bar_x, bar_y, bar_width, bar_height),
            border_radius=3,
        )

        center_x = bar_x + bar_width // 2
        pygame.draw.line(
            screen,
            TITLE_PINK,
            (center_x, bar_y - 6),
            (center_x, bar_y + bar_height + 6),
            2,
        )

        if has_pitch:
            max_cents = 50.0
            c = max(-max_cents, min(max_cents, cents))
            t = (c + max_cents) / (2 * max_cents)  # 0..1
            marker_x = int(bar_x + t * bar_width)

            pygame.draw.circle(
                screen,
                key_color,
                (marker_x, bar_y + bar_height // 2),
                7,
            )

            left_hint = small_font.render("flat", True, MUTED_TEXT)
            right_hint = small_font.render("sharp", True, MUTED_TEXT)
            screen.blit(left_hint, (bar_x, bar_y + 12))
            screen.blit(
                right_hint,
                (bar_x + bar_width - right_hint.get_width(), bar_y + 12),
            )

        # --- Info + practice feedback ---
        info = info_font.render(msg, True, TEXT_COLOR)
        screen.blit(
            info,
            (piano_rect.x + PIANO_PADDING_X, bar_y + 34),
        )

        if practice.last_result:
            last_text = small_font.render(
                f"Last: {practice.last_result}", True, MUTED_TEXT
            )
            screen.blit(
                last_text,
                (piano_rect.x + PIANO_PADDING_X, bar_y + 60),
            )

        controls_text = small_font.render(
            "Click keys to hear them  •  ←/→ note  •  ↑/↓ octave  •  R reset",
            True,
            MUTED_TEXT,
        )
        screen.blit(
            controls_text,
            (
                piano_rect.x + PIANO_PADDING_X,
                piano_rect.bottom - controls_text.get_height() - 8,
            ),
        )

        pygame.display.flip()

    stream.stop()
    stream.close()
    cap.release()
    pygame.quit()


if __name__ == "__main__":
    run_app()

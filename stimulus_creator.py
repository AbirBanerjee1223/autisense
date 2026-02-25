# stimulus_creator.py
# Run once: python stimulus_creator.py
# Generates all stimulus files in stimuli/ folder

import cv2
import numpy as np
import wave
import struct
import math
import time
from pathlib import Path
from config import STIMULI_DIR

def create_social_geometric_video():
    """
    Create a split-screen stimulus video:
    LEFT side: Animated "social" stimulus (face-like pattern
               with eyes that blink, mouth that moves)
    RIGHT side: Animated "geometric" stimulus (spinning 
                shapes, moving patterns)
    
    Duration: 30 seconds at 24 fps
    """
    print("🎬 Creating social vs geometric stimulus video...")

    output_path = str(STIMULI_DIR / "social_geometric.mp4")
    width, height = 960, 480  # Split: 480x480 each side
    fps = 24
    duration = 30  # seconds
    total_frames = fps * duration

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(total_frames):
        t = frame_idx / fps  # Time in seconds
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # ===== LEFT SIDE: SOCIAL STIMULUS =====
        left = canvas[:, :480]

        # Warm background
        left[:] = (220, 200, 180)

        # Face circle (skin tone)
        face_cx, face_cy = 240, 200
        face_radius = 120
        # Gentle bobbing motion
        bob_y = int(8 * math.sin(t * 1.5))
        cv2.circle(
            left, (face_cx, face_cy + bob_y),
            face_radius, (180, 210, 230), -1
        )
        cv2.circle(
            left, (face_cx, face_cy + bob_y),
            face_radius, (150, 180, 200), 2
        )

        # Eyes
        eye_y = face_cy - 20 + bob_y
        # Blinking animation (blink every 3 seconds)
        blink_phase = (t % 3.0)
        if blink_phase < 0.15:
            eye_height = 3  # Closed
        else:
            eye_height = 18  # Open

        # Left eye
        cv2.ellipse(
            left, (200, eye_y), (20, eye_height),
            0, 0, 360, (80, 60, 40), -1
        )
        # Right eye
        cv2.ellipse(
            left, (280, eye_y), (20, eye_height),
            0, 0, 360, (80, 60, 40), -1
        )

        # Pupils (follow a pattern - look at camera mostly)
        pupil_offset_x = int(5 * math.sin(t * 0.8))
        if eye_height > 5:
            cv2.circle(
                left, (200 + pupil_offset_x, eye_y),
                8, (30, 20, 10), -1
            )
            cv2.circle(
                left, (280 + pupil_offset_x, eye_y),
                8, (30, 20, 10), -1
            )

        # Mouth (smiling animation)
        mouth_y = face_cy + 50 + bob_y
        smile_amount = int(15 * abs(math.sin(t * 1.2)))
        cv2.ellipse(
            left, (face_cx, mouth_y),
            (35, 10 + smile_amount),
            0, 0, 180, (100, 80, 120), 3
        )

        # Waving hand
        hand_x = 380
        hand_y = 280 + int(40 * math.sin(t * 3.0))
        cv2.circle(left, (hand_x, hand_y), 25, (180, 210, 230), -1)
        # Fingers
        for finger_angle in [-30, -10, 10, 30]:
            rad = math.radians(finger_angle + 20 * math.sin(t * 4))
            fx = hand_x + int(20 * math.sin(rad))
            fy = hand_y - int(25 * math.cos(rad))
            cv2.line(
                left, (hand_x, hand_y - 15),
                (fx, fy), (180, 210, 230), 4
            )

        # Label
        cv2.putText(
            left, "SOCIAL",
            (180, 460), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (100, 100, 100), 2
        )

        # ===== RIGHT SIDE: GEOMETRIC STIMULUS =====
        right = canvas[:, 480:]

        # Dark background
        right[:] = (30, 30, 50)

        # Spinning triangles
        for i in range(3):
            angle = t * (2.0 + i * 0.5) + i * 2.09
            cx = 240 + int(80 * math.cos(angle * 0.3))
            cy = 200 + int(80 * math.sin(angle * 0.3))
            size = 40 + i * 15

            pts = []
            for j in range(3):
                a = angle + j * (2 * math.pi / 3)
                px = cx + int(size * math.cos(a))
                py = cy + int(size * math.sin(a))
                pts.append([px, py])

            pts_np = np.array(pts, dtype=np.int32)
            colors = [
                (255, 100, 100),
                (100, 255, 100),
                (100, 100, 255)
            ]
            cv2.fillPoly(right, [pts_np], colors[i])

        # Spinning circles
        for i in range(5):
            angle = t * 3.0 + i * 1.26
            cx = 240 + int(150 * math.cos(angle))
            cy = 240 + int(150 * math.sin(angle))
            radius = 15 + int(10 * math.sin(t * 5 + i))
            color = (
                int(127 + 127 * math.sin(t * 2 + i)),
                int(127 + 127 * math.sin(t * 2 + i + 2)),
                int(127 + 127 * math.sin(t * 2 + i + 4))
            )
            cv2.circle(right, (cx, cy), radius, color, -1)

        # Pulsing central shape
        pulse = 30 + int(20 * math.sin(t * 4))
        cv2.rectangle(
            right,
            (240 - pulse, 200 - pulse),
            (240 + pulse, 200 + pulse),
            (200, 200, 50), 3
        )

        # Moving lines
        for i in range(4):
            x_start = int((t * 100 + i * 120) % 480)
            cv2.line(
                right, (x_start, 0), (480 - x_start, 480),
                (80, 80, 120), 1
            )

        cv2.putText(
            right, "GEOMETRIC",
            (160, 460), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (150, 150, 150), 2
        )

        # ===== CENTER DIVIDER =====
        cv2.line(canvas, (480, 0), (480, height), (255, 255, 255), 2)

        writer.write(canvas)

    writer.release()
    print(f"  ✅ Saved: {output_path}")
    return output_path


def create_smile_prompt_video():
    """
    Create a video that shows a person smiling broadly,
    used to test emotional reciprocity.
    
    Shows a face that transitions from neutral to big smile.
    Duration: 10 seconds
    """
    print("🎬 Creating smile prompt video...")

    output_path = str(STIMULI_DIR / "smile_prompt.mp4")
    width, height = 480, 480
    fps = 24
    duration = 10
    total_frames = fps * duration

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 240

        face_cx, face_cy = 240, 220
        face_r = 140

        # Face
        cv2.circle(canvas, (face_cx, face_cy), face_r, (180, 210, 230), -1)
        cv2.circle(canvas, (face_cx, face_cy), face_r, (150, 180, 200), 2)

        # Eyes (always open, friendly)
        eye_y = face_cy - 30
        cv2.ellipse(canvas, (190, eye_y), (22, 20), 0, 0, 360, (80, 60, 40), -1)
        cv2.ellipse(canvas, (290, eye_y), (22, 20), 0, 0, 360, (80, 60, 40), -1)

        # Pupils
        cv2.circle(canvas, (190, eye_y), 9, (30, 20, 10), -1)
        cv2.circle(canvas, (290, eye_y), 9, (30, 20, 10), -1)

        # Eye shine
        cv2.circle(canvas, (194, eye_y - 4), 3, (255, 255, 255), -1)
        cv2.circle(canvas, (294, eye_y - 4), 3, (255, 255, 255), -1)

        # Smile progression
        # First 3 seconds: neutral, then transition to smile
        if t < 3.0:
            smile = 0.0
            label = "Watch the face..."
        elif t < 5.0:
            smile = (t - 3.0) / 2.0  # 0 to 1 over 2 seconds
            label = ""
        else:
            smile = 1.0
            label = ""

        mouth_y = face_cy + 55
        mouth_width = int(30 + 25 * smile)
        mouth_curve = int(5 + 30 * smile)

        if smile < 0.3:
            # Neutral mouth
            cv2.ellipse(
                canvas, (face_cx, mouth_y),
                (mouth_width, 5), 0, 0, 180,
                (100, 80, 120), 3
            )
        else:
            # Smiling mouth
            cv2.ellipse(
                canvas, (face_cx, mouth_y),
                (mouth_width, mouth_curve),
                0, 0, 180, (100, 80, 120), 3
            )
            # Cheek circles (blush)
            blush_alpha = smile * 0.5
            cv2.circle(
                canvas, (155, face_cy + 25),
                20, (180, 190, 230), -1
            )
            cv2.circle(
                canvas, (325, face_cy + 25),
                20, (180, 190, 230), -1
            )

        # Eyebrows raise with smile
        brow_raise = int(10 * smile)
        cv2.ellipse(
            canvas, (190, eye_y - 35 - brow_raise),
            (28, 8), 0, 180, 360, (120, 100, 80), 2
        )
        cv2.ellipse(
            canvas, (290, eye_y - 35 - brow_raise),
            (28, 8), 0, 180, 360, (120, 100, 80), 2
        )

        if label:
            cv2.putText(
                canvas, label,
                (120, 430), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (100, 100, 100), 2
            )

        if t >= 5.0:
            cv2.putText(
                canvas, "Can you smile too?",
                (100, 430), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (60, 120, 60), 2
            )

        writer.write(canvas)

    writer.release()
    print(f"  ✅ Saved: {output_path}")
    return output_path


def create_name_call_audio():
    """
    Create a clear audio stimulus: two claps followed
    by "Hey!" tone burst.
    
    Using pure sine wave synthesis (no external files needed).
    """
    print("🔊 Creating name-call audio stimulus...")

    output_path = str(STIMULI_DIR / "name_call.wav")
    sample_rate = 44100
    
    def silence(duration_sec):
        return [0] * int(sample_rate * duration_sec)
    
    def tone_burst(freq, duration_sec, volume=0.8):
        """Generate a tone burst with attack/decay envelope."""
        n_samples = int(sample_rate * duration_sec)
        samples = []
        for i in range(n_samples):
            t = i / sample_rate
            # Envelope: quick attack, gradual decay
            envelope = min(t * 20, 1.0) * max(0, 1.0 - t / duration_sec)
            value = volume * envelope * math.sin(2 * math.pi * freq * t)
            samples.append(value)
        return samples
    
    def clap_sound(duration_sec=0.05, volume=0.9):
        """Generate a clap-like noise burst."""
        n_samples = int(sample_rate * duration_sec)
        samples = []
        import random
        random.seed(42)
        for i in range(n_samples):
            t = i / n_samples
            envelope = max(0, 1.0 - t * 3)  # Sharp decay
            value = volume * envelope * (random.random() * 2 - 1)
            samples.append(value)
        return samples

    # Build the audio sequence:
    # 3 seconds silence -> CLAP -> 0.3s -> CLAP -> 0.5s -> "HEY" tone -> silence
    audio = []
    audio.extend(silence(3.0))
    audio.extend(clap_sound(0.08))
    audio.extend(silence(0.3))
    audio.extend(clap_sound(0.08))
    audio.extend(silence(0.5))

    # "Hey" approximation: rising frequency burst
    hey_duration = 0.4
    n_hey = int(sample_rate * hey_duration)
    for i in range(n_hey):
        t = i / sample_rate
        freq = 300 + 400 * (t / hey_duration)  # Rising pitch
        envelope = min(t * 15, 1.0) * max(0, 1.0 - (t / hey_duration) ** 2)
        value = 0.8 * envelope * math.sin(2 * math.pi * freq * t)
        audio.append(value)

    audio.extend(silence(3.0))

    # Write WAV file
    n_frames = len(audio)
    with wave.open(output_path, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        for sample in audio:
            clamped = max(-1.0, min(1.0, sample))
            packed = struct.pack('h', int(clamped * 32767))
            wav.writeframes(packed)

    print(f"  ✅ Saved: {output_path}")
    print(f"     Duration: {n_frames / sample_rate:.1f}s")
    print(f"     Audio cue at: ~3.0 seconds")
    return output_path


def create_all_stimuli():
    """Generate all stimulus files."""
    print("=" * 50)
    print("  STIMULUS GENERATOR")
    print("=" * 50)

    create_social_geometric_video()
    create_smile_prompt_video()
    create_name_call_audio()

    print("\n✅ All stimuli generated in stimuli/ folder")
    print("   Ready for screening sessions.")


if __name__ == "__main__":
    create_all_stimuli()
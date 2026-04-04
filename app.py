"""
From Face to Form

author: manmeet sagri | iat 460

---

About this app :
    this system takes a video clip as input and transforms facial emotion data
    into two evolving generative art outputs: a julia fractal and an l-system tree.
    both structures are driven by the same probabilistic emotion signal, so as the
    emotional tone of the scene shifts, both visuals respond in real time.

the pipeline works in five stages:

    1. frame sampling
       the uploaded video is sampled at a fixed interval (every 0.2s by default)
       rather than processing every single frame. this keeps the computation
       feasible while still capturing how emotion changes across the scene.

    2. face detection
       each sampled frame is scanned using opencv's haar cascade classifier to
       find the largest face. the detected face crop is passed to the emotion model.

    3. emotion classification
       the face crop is fed into a pretrained vision transformer (vit) model
       — trpakov/vit-face-expression — which returns a probability distribution
       across 7 emotion categories: happy, sad, fear, angry, surprise, neutral, disgust.
       these aren't binary labels, they're weighted scores that sum to 1.

    4. temporal smoothing
       raw frame-by-frame emotion values can jump around a lot. exponential
       moving average smoothing is applied to create a more stable, continuous
       signal — so the generative output evolves gradually rather than flickering.

    5. generative output
       the smoothed emotion probabilities are mapped to visual parameters:
         - julia fractal: the constant c is shifted, zoom and turbulence are modulated,
           color is blended from an emotion palette, and iteration depth varies.
         - l-system tree: recursion depth, branching angle, step length, and production
           rules all respond to which emotions are active and at what intensity.
       both outputs are rendered side by side into a composite video.

development notes:
    - started with just the julia fractal, then added the l-system layer
      after confirming the emotion pipeline was producing stable outputs.
    - the emotion-to-parameter mappings were tuned manually by testing with
      clips that had clear, readable emotional content.
    - the haar cascade was chosen over a deep learning face detector because
      it's fast enough to run on cpu without throttling the rest of the pipeline.
    - the 8000-character cap on l-system strings was added after discovering
      that anger rules at depth 5+ could produce strings long enough to freeze rendering.
    - the composite video runs at 5fps intentionally — slow enough to read
      the emotional shifts, fast enough to feel like motion.
"""

import json
import math
import tempfile

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from transformers import pipeline


# =========================================================
# custom ui styling
# =========================================================

# all visual styling for the gradio interface lives here as a css string.
# it gets passed into gr.Blocks() at launch time via the css= argument.
# keeping it in one place makes it easy to update the look without
# digging through the layout code.
CUSTOM_CSS = """
body {
    background: #0b0b0f;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding-top: 48px !important;
    padding-bottom: 60px !important;
}

.main-title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
    color: #f5f5f7;
}

.subtitle {
    text-align: center;
    font-size: 1.05rem;
    color: #b9bcc6;
    max-width: 650px;
    margin: 0 auto 2.5rem auto;
    line-height: 1.65;
}

.section-heading {
    text-align: center;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f1f3;
    margin-top: 0;
    margin-bottom: 12px;
}

.info-text {
    text-align: center;
    color: #c9ccd4;
    font-size: 0.95rem;
    line-height: 1.55;
}

.analysis-shell {
    background: #13151c;
    border: 1px solid #2a2d36;
    border-radius: 18px;
    padding: 16px;
    margin-top: 12px;
}

.video-panel {
    background: linear-gradient(180deg, #161821 0%, #111319 100%);
    border: 1px solid #2a2d36;
    border-radius: 18px;
    padding: 14px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.28);
}

.footer-note {
    text-align: center;
    color: #8f95a3;
    font-size: 0.88rem;
    margin-top: 40px;
    padding-top: 24px;
    border-top: 1px solid #1e2028;
}

/* White primary button */
button.primary,
.gradio-container button[variant="primary"] {
    background: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    transition: opacity 0.15s ease !important;
}
button.primary:hover {
    opacity: 0.88 !important;
}

/* Section spacing */
.section-gap {
    margin-top: 40px;
}
"""


##############################
# model and detector loading
##############################

# opencv's haar cascade is a classical, fast face detector.
# it works by scanning the image at multiple scales for patterns
# that match a pre-trained frontal face template.
# we use it here because it's lightweight enough to run per-frame on cpu.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# this loads the pretrained vision transformer (vit) model from hugging face.
# the model was trained on facial expression data and returns probability scores
# for each emotion category. it runs once at startup and is reused across all frames.
emotion_classifier = pipeline(
    "image-classification",
    model="trpakov/vit-face-expression"
)


###############################
# video processing utilities
###############################

def resolve_video_path(video_file):
    # gradio can pass the video in a few different formats depending on the version 
    # sometimes it's a plain string path, sometimes it's a dict with a "path" or "video" key.
    # this function normalizes all of those cases into a single usable file path string.
    if video_file is None:
        return None
    if isinstance(video_file, str):
        return video_file
    if isinstance(video_file, dict):
        if "path" in video_file:
            return video_file["path"]
        if "video" in video_file:
            return video_file["video"]
    return None


def sample_frames(
    video_path: str,
    seconds_between_samples: float = 0.2,
    max_frames: int = 60
) -> list[np.ndarray]:
    # instead of processing every frame (which would be very slow),
    # we sample one frame every 0.2 seconds worth of video.
    # the interval is calculated from the video's actual fps so it stays consistent
    # regardless of whether the clip is 24fps, 30fps, or something else.
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 24.0  # fall back to a safe default if fps metadata is missing
    sample_every = max(1, int(fps * seconds_between_samples))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            # opencv reads frames as bgr by default — convert to rgb for the rest of the pipeline
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            if len(frames) >= max_frames:
                break
        frame_idx += 1
    cap.release()
    return frames


###############################
# face detection utilities
###############################

def detect_face_with_box(frame: np.ndarray):
    # run the haar cascade on a grayscale version of the frame 
    # the detector operates on grayscale intensity, not color.
    # scaleFactor controls how much the image is downscaled at each scan step.
    # minNeighbors filters out weak detections... higher = stricter.
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    # if multiple faces are found, take the largest one by area.
    # this targets the primary subject in cinematic footage.
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_crop = frame[y:y + h, x:x + w]
    return face_crop, (x, y, w, h)


def draw_face_box(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    # draws a bounding box rectangle onto a copy of the frame for display purposes.
    # operates on a copy so the original frame data isn't modified.
    display_frame = frame.copy()
    x, y, w, h = box
    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 90, 90), 2)
    return display_frame


###############################
# emotion processing utilities
###############################

def normalize_label(label: str) -> str:
    # the vit model uses label names like "anger" and "happiness" (full words),
    # while the rest of the pipeline uses short forms like "angry" and "happy".
    # this maps all variations to a consistent set of keys.
    label = label.strip().lower()
    label_map = {
        "anger": "angry", "angry": "angry",
        "sadness": "sad", "sad": "sad",
        "happiness": "happy", "happy": "happy",
        "fear": "fear", "surprise": "surprise",
        "neutral": "neutral", "disgust": "disgust",
    }
    return label_map.get(label, label)


def results_to_dict(result: list[dict]) -> dict[str, float]:
    # the hugging face pipeline returns a list of {"label": ..., "score": ...} dicts.
    # this converts that into a clean {emotion: probability} mapping
    # with normalized label names.
    normalized = {}
    for item in result:
        label = normalize_label(item["label"])
        normalized[label] = float(item["score"])
    return normalized


def smooth_emotion_sequence(
    emotion_sequence: list[dict[str, float]],
    alpha: float = 0.2
) -> list[dict[str, float]]:
    # applies exponential moving average (ema) smoothing to the emotion sequence.
    # without smoothing, the per-frame probabilities can jump sharply between frames,
    # which would make the generative output flicker rather than evolve smoothly.
    # alpha controls how much weight the previous value carries... lower = smoother.
    if not emotion_sequence:
        return []
    keys = set()
    for emotions in emotion_sequence:
        keys.update(emotions.keys())
    smoothed = []
    prev = {k: 0.0 for k in keys}
    for i, emotions in enumerate(emotion_sequence):
        current = {k: emotions.get(k, 0.0) for k in keys}
        if i == 0:
            prev = current.copy()
        else:
            prev = {
                k: alpha * prev.get(k, 0.0) + (1 - alpha) * current.get(k, 0.0)
                for k in keys
            }
        smoothed.append(prev.copy())
    return smoothed


def average_emotions(emotion_sequence: list[dict[str, float]]) -> dict[str, float]:
    # computes a single averaged emotion dict across all frames 
    # used to determine the dominant emotional tone of the whole clip
    # for the summary display.
    if not emotion_sequence:
        return {}
    keys = set()
    for emotions in emotion_sequence:
        keys.update(emotions.keys())
    averaged = {}
    for key in keys:
        averaged[key] = float(np.mean([e.get(key, 0.0) for e in emotion_sequence]))
    return averaged


def format_top_emotions_from_dict(emotions: dict[str, float], top_k: int = 3) -> str:
    # builds an html string for displaying the top-k emotions as styled progress bars.
    # each bar is color-coded to match the emotion's palette color used in the fractals.
    if not emotions:
        return "<p style='color:#888'>No emotions detected.</p>"

    color_map = {
        "happy":   "#FFBE3C",
        "sad":     "#5A82FF",
        "fear":    "#AAF5FF",
        "angry":   "#FF4646",
        "surprise":"#FF69B4",
        "neutral": "#A0A0A0",
        "disgust": "#82B46E",
    }

    ranked = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:top_k]

    pills = ""
    for label, score in ranked:
        color = color_map.get(label, "#cccccc")
        pct = int(score * 100)
        pills += f"""
        <div style="margin-bottom:12px;">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
            <div style="display:flex;align-items:center;gap:8px;">
              <div style="width:14px;height:14px;border-radius:4px;background:{color};flex-shrink:0;"></div>
              <span style="color:#f1f1f3;font-size:1rem;font-weight:700;text-transform:capitalize;">{label}</span>
            </div>
            <span style="color:{color};font-size:0.95rem;font-weight:700;">{score:.1%}</span>
          </div>
          <div style="background:#1e1e2e;border-radius:6px;height:10px;overflow:hidden;">
            <div style="background:{color};width:{pct}%;height:100%;border-radius:6px;transition:width 0.3s;"></div>
          </div>
        </div>
        """

    return f"<div style='padding:8px 0;'>{pills}</div>"


def get_emotion_legend_text() -> str:
    # returns a plain text summary of how each emotion maps to visual parameters.
    # shown in the "view more details" accordion so the viewer understands
    # the logic behind each visual output.
    return (
        "Julia Fractal\n"
        "Happy → warm gold/orange, bright and expansive\n"
        "Sad → cool blue/violet, withdrawn and dim\n"
        "Fear → icy cyan, irregular and unstable\n"
        "Angry → red/crimson, dense and forceful\n"
        "Surprise → hot-pink, sudden expansion\n\n"
        "L-System Tree\n"
        "Happy → tall dense tree, warm stroke, many iterations\n"
        "Sad → sparse narrow tree, cool stroke, fewer iterations\n"
        "Fear → erratic branching angles, high asymmetry\n"
        "Angry → wide aggressive spread, fast growth\n"
        "Surprise → sudden depth jump, sharp angle shift"
    )


###############################
# julia fractal generation
###############################

def blend_emotion_color(emotions: dict[str, float]) -> np.ndarray:
    # each emotion is assigned a base rgb color.
    # the final color is a weighted blend of all active emotions,
    # proportional to their probability scores.
    # this ensures the full distribution influences the color
    # not just the dominant emotion.
    palette = {
        "happy":   np.array([255, 190,  60], dtype=np.float32),
        "sad":     np.array([ 90, 130, 255], dtype=np.float32),
        "fear":    np.array([170, 245, 255], dtype=np.float32),
        "angry":   np.array([255,  70,  70], dtype=np.float32),
        "surprise":np.array([255,  20, 147], dtype=np.float32),
        "neutral": np.array([185, 185, 185], dtype=np.float32),
        "disgust": np.array([130, 180, 110], dtype=np.float32),
    }
    total = sum(emotions.values())
    if total <= 0:
        return np.array([190, 190, 190], dtype=np.float32)
    color = np.zeros(3, dtype=np.float32)
    for label, score in emotions.items():
        if label in palette:
            color += palette[label] * score
    color = color / total
    return np.clip(color, 0, 255)


def generate_julia(emotions: dict[str, float], width: int = 360, height: int = 360) -> np.ndarray:
    # the julia set is defined by iterating z → z² + c for each point in the complex plane.
    # the constant c determines the shape of the fractal.. different values of c
    # produce dramatically different structures. here, c is shifted by emotion values
    # so the fractal's form changes with the emotional state.
    happy   = emotions.get("happy",   0.0)
    sad     = emotions.get("sad",     0.0)
    fear    = emotions.get("fear",    0.0)
    angry   = emotions.get("angry",   0.0)
    surprise= emotions.get("surprise",0.0)
    neutral = emotions.get("neutral", 0.0)
    disgust = emotions.get("disgust", 0.0)

    # c controls the fractal's shape.. fear pushes it toward more chaotic territory,
    # sadness pulls back toward simpler forms, surprise adds imaginary component variation.
    c_real = -0.72 + fear * 0.90 - sad * 0.45 + happy * 0.30 - angry * 0.30
    c_imag =  0.18 + surprise * 0.80 - angry * 0.45 + neutral * 0.05 + disgust * 0.15

    # more iterations = more detail and complexity in the boundary regions.
    # high-energy emotions (angry, happy, surprise) push this up.
    max_iter = int(90 + happy * 80 + angry * 90 + surprise * 70 + fear * 65 + disgust * 35)

    # zoom controls how much of the complex plane is visible.
    # sadness zooms out (retreating), happiness zooms in slightly (expansive).
    zoom = 1.0 + sad * 0.85 + fear * 0.25 - happy * 0.20 - surprise * 0.12

    # turbulence adds a sinusoidal warp to the grid before iteration —
    # fear and anger introduce instability, making the fractal look unsettled.
    turbulence = fear * 0.10 + angry * 0.08 + surprise * 0.06 + disgust * 0.04

    x = np.linspace(-1.5 * zoom, 1.5 * zoom, width)
    y = np.linspace(-1.5 * zoom, 1.5 * zoom, height)
    X, Y = np.meshgrid(x, y)

    if turbulence > 0:
        X = X + turbulence * np.sin(6 * Y)
        Y = Y + turbulence * np.cos(6 * X)

    Z = X + 1j * Y
    C = complex(c_real, c_imag)

    # escape-time algorithm: track how many iterations each point takes to escape radius 2.
    # points that never escape are inside the julia set (rendered as solid).
    # the escape count becomes the brightness value for each pixel.
    fractal = np.zeros(Z.shape, dtype=np.float32)
    escaped = np.zeros(Z.shape, dtype=bool)

    for i in range(max_iter):
        Z = Z ** 2 + C
        mask = (np.abs(Z) > 2.0) & (~escaped)
        fractal[mask] = i
        escaped[mask] = True

    # points that never escaped get the maximum value (interior of the set)
    fractal[~escaped] = max_iter
    if fractal.max() > 0:
        fractal = fractal / fractal.max()

    # apply gamma correction to boost contrast.. angry/fear compress it, sadness softens it
    contrast_power = max(0.35, 0.65 - angry * 0.10 - fear * 0.08 + sad * 0.05)
    fractal = fractal ** contrast_power

    # overall brightness: happiness and surprise make it brighter, sadness dims it
    brightness = 0.55 + happy * 0.18 + surprise * 0.20 - sad * 0.08
    fractal = np.clip(fractal * brightness, 0, 1)

    # apply the blended emotion color across all three rgb channels
    color = blend_emotion_color(emotions) / 255.0
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(3):
        rgb[:, :, i] = fractal * color[i]

    # add a subtle glow layer to the interior.. neutral emotion adds a soft luminance
    glow = np.clip(fractal * (0.10 + neutral * 0.05), 0, 1)
    rgb += glow[:, :, None] * 0.08
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


###############################
# l-system generation (course week 3 lab concept)
###############################

def build_lsystem_string(axiom: str, rules: dict[str, str], iterations: int) -> str:
    """
    Expand the axiom by applying production rules for a given number of iterations.
    Directly mirrors the create_l_system function from the course L-systems lab.
    """
    # start from the axiom and repeatedly apply the production rules.
    # each character in the string gets replaced by its rule expansion.
    # characters with no rule stay as is.
    # this mirrors the rewriting system from the week 3 l-systems lab.
    result = axiom
    for _ in range(iterations):
        new_string = ""
        for char in result:
            new_string += rules.get(char, char)
        result = new_string
    return result


def draw_lsystem_to_image(
    instructions: str,
    angle_deg: float,
    step_len: float,
    stroke_color: tuple[int, int, int],
    width: int = 360,
    height: int = 360,
    bg_color: tuple[int, int, int] = (10, 10, 18),
) -> np.ndarray:
    """
    Render an L-system string to an RGB image using a stack-based turtle interpreter.
    Uses the same command set taught in the course lab:
      F = draw forward, + = turn right, - = turn left, [ = push, ] = pop
    """
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Start near the bottom center, pointing up
    x, y = width / 2, height * 0.88
    heading = 90.0  # degrees, measured from positive x-axis (90 = up)
    stack = []

    angle_rad = math.radians(angle_deg)

    # turtle interpreter: each character in the l-system string is a drawing command.
    # F = move forward and draw a line
    # + / - = rotate left or right by the current angle
    # [ / ] = push or pop the turtle's position and heading onto a stack
    #         this is what allows branching — save state, draw a branch, restore state
    for cmd in instructions:
        if cmd == "F":
            nx = x + step_len * math.cos(math.radians(heading))
            ny = y - step_len * math.sin(math.radians(heading))
            draw.line([(x, y), (nx, ny)], fill=stroke_color, width=1)
            x, y = nx, ny
        elif cmd == "+":
            heading -= angle_deg
        elif cmd == "-":
            heading += angle_deg
        elif cmd == "[":
            stack.append((x, y, heading))
        elif cmd == "]":
            if stack:
                x, y, heading = stack.pop()

    return np.array(img)


def get_lsystem_stroke_color(emotions: dict[str, float]) -> tuple[int, int, int]:
    """
    Map emotion blend to an L-system stroke color, distinct from the Julia palette.
    Warm tones for positive emotions, cool/muted for negative.
    """
    # the l-system gets a separate color from the julia fractal
    # so the two panels are visually distinct even when driven by the same emotion data.
    # r channel is boosted by high-energy emotions (happy, angry, surprise),
    # b channel is boosted by low-energy emotions (sad, fear).
    happy   = emotions.get("happy",   0.0)
    sad     = emotions.get("sad",     0.0)
    fear    = emotions.get("fear",    0.0)
    angry   = emotions.get("angry",   0.0)
    surprise= emotions.get("surprise",0.0)
    neutral = emotions.get("neutral", 0.0)

    r = int(np.clip(80  + happy * 160 + angry * 140 + surprise * 100, 0, 255))
    g = int(np.clip(140 + happy * 80  - angry * 60  + neutral * 40,   0, 255))
    b = int(np.clip(200 + sad * 55    + fear * 55   - happy * 80,     0, 255))
    return (r, g, b)


def generate_lsystem(emotions: dict[str, float], width: int = 360, height: int = 360) -> np.ndarray:
    """
    Generate an emotion-driven L-system tree image.

    Emotion-to-parameter mapping (from project proposal):
      Happy   → more iterations (denser tree), wider angle, longer step
      Sad     → fewer iterations, narrower angle, shorter step
      Fear    → high angle variability (asymmetric rules), moderate iterations
      Angry   → aggressive spread, wider angle, more branching density
      Surprise→ sudden depth jump, sharp angle shift
    """
    happy   = emotions.get("happy",   0.0)
    sad     = emotions.get("sad",     0.0)
    fear    = emotions.get("fear",    0.0)
    angry   = emotions.get("angry",   0.0)
    surprise= emotions.get("surprise",0.0)
    neutral = emotions.get("neutral", 0.0)

    # recursion depth: happy/surprise push it up, sad pulls it down
    # higher iterations = more branching complexity, but also longer render time
    iterations = int(np.clip(
        3 + round(happy * 2) + round(surprise * 2) - round(sad * 1.5) + round(angry * 1),
        2, 6
    ))

    # branching angle: fear and anger widen it, sadness narrows
    base_angle = 25.0
    angle = base_angle + angry * 20 + fear * 15 + surprise * 12 - sad * 10 + happy * 8
    angle = float(np.clip(angle, 10, 55))

    # step length: happy = taller, sad = shorter
    step = float(np.clip(8 + happy * 5 - sad * 3 + angry * 2, 3, 16))

    # Fear adds asymmetry by shifting the right-branch angle slightly
    fear_skew = fear * 8.0

    # choose production rules based on the dominant emotional state.
    # each emotion category gets a different branching grammar
    # that produces a visually distinct tree structure.
    if fear > 0.35:
        # irregular, erratic branching (fear: high angle variability)
        right_angle = angle + fear_skew
        left_angle  = angle - fear_skew * 0.5
        axiom = "X"
        rules = {
            "X": f"F+[[X]-X]-F[-FX]+X",
            "F": "FF",
        }
    elif angry > 0.35:
        # dense aggressive spread (anger: fast expansion, dense clusters)
        axiom = "F"
        rules = {
            "F": "F[+F]F[-F][F]F[+F][-F]",
        }
    elif sad > 0.35:
        # sparse, drooping structure (sadness: minimal branching)
        axiom = "F"
        rules = {
            "F": "F[-F]F",
        }
    else:
        # default balanced plant (happy/neutral/surprise)
        axiom = "X"
        rules = {
            "X": "F+[[X]-X]-F[-FX]+X",
            "F": "FF",
        }

    instructions = build_lsystem_string(axiom, rules, iterations)

    # cap instruction length to avoid rendering timeouts 
    # especially the anger rule which expands very aggressively at higher depths.
    instructions = instructions[:8000]

    stroke = get_lsystem_stroke_color(emotions)
    return draw_lsystem_to_image(instructions, angle, step, stroke, width, height)


###############################
# composite frame: julia (left) | l-system (right)
###############################

def make_composite_frame(
    julia_img: np.ndarray,
    lsystem_img: np.ndarray,
    panel_size: int = 360,
) -> np.ndarray:
    """
    Stack the Julia fractal and L-system tree side by side with a thin divider.
    Both panels are padded/labeled so the composite reads as a unified artwork.
    """
    divider_w = 4
    label_h = 22
    total_w = panel_size * 2 + divider_w
    total_h = panel_size + label_h

    # create a dark canvas and place both panels into it
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:, :, :] = [14, 14, 22]  # dark background

    # Place Julia on the left
    canvas[label_h:label_h + panel_size, :panel_size] = julia_img

    # thin vertical divider between the two panels
    canvas[label_h:, panel_size:panel_size + divider_w] = [40, 40, 55]

    # Place L-system on the right
    canvas[label_h:label_h + panel_size, panel_size + divider_w:] = lsystem_img

    # Draw labels inside the composite video frame.. bigger and with shadow
    font = cv2.FONT_HERSHEY_SIMPLEX
    for text, tx in [("Julia Fractal", 10), ("L-System Tree", panel_size + divider_w + 10)]:
        cv2.putText(canvas, text, (tx + 1, 19), font, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, text, (tx, 18), font, 0.65, (200, 200, 230), 1, cv2.LINE_AA)

    return canvas


###############################
# sample strip generation (frame | emotions | fractal)
###############################

def make_emotion_bar_image(
    emotions: dict[str, float],
    width: int = 500,
    height: int = 400,
) -> np.ndarray:
    """
    Draw a vertical bar chart of emotion probabilities using PIL.
    """
    # renders a simple horizontal bar chart for each emotion,
    # color-coded to match the system's emotion palette.
    # used in the sample strip breakdown to show what the model actually detected.
    from PIL import ImageFont
    img = Image.new("RGB", (width, height), (12, 12, 22))
    draw = ImageDraw.Draw(img)

    palette = {
        "happy":   (255, 190,  60),
        "sad":     ( 90, 130, 255),
        "fear":    (170, 245, 255),
        "angry":   (255,  70,  70),
        "surprise":(180,  60, 180),
        "neutral": (160, 160, 160),
        "disgust": (130, 180, 110),
    }

    if not emotions:
        return np.array(img)

    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_emotions)
    pad = 16
    bar_area_w = width - pad * 2
    bar_h = max(28, (height - pad * 2 - (n - 1) * 10) // n)

    for i, (label, score) in enumerate(sorted_emotions):
        y = pad + i * (bar_h + 10)
        bar_w = int(bar_area_w * score)
        color = palette.get(label, (180, 180, 180))

        # Background track
        draw.rectangle([pad, y, pad + bar_area_w, y + bar_h], fill=(30, 30, 45))
        # Filled bar
        if bar_w > 0:
            draw.rectangle([pad, y, pad + bar_w, y + bar_h], fill=color)

        # Label and score — large text with shadow
        label_text = f"{label}: {score:.2f}"
        draw.text((pad + 6 + 1, y + 5 + 1), label_text, fill=(0, 0, 0))
        draw.text((pad + 6,     y + 5),     label_text, fill=(255, 255, 255))

    return np.array(img)


def build_sample_strip(
    detected_entries: list[dict],
    smoothed_emotions: list[dict[str, float]],
    composite_frames: list[np.ndarray],
    n_samples: int = 8,
) -> list[np.ndarray]:
    """
    Build a list of contact-sheet images, each showing:
      face crop | emotion bars | composite fractal frame
    Evenly spaced across the detected sequence.
    """
    # creates a gallery of evenly spaced sample "strips" across the full clip.
    # each strip shows the face crop, the emotion probability bars, and
    # the corresponding generative composite frame side by side.
    # this gives a readable snapshot of how the system evolved over time.
    from PIL import ImageFont
    total = len(detected_entries)
    if total == 0:
        return []

    indices = [int(i * (total - 1) / (n_samples - 1)) for i in range(n_samples)] if total >= n_samples else list(range(total))
    indices = list(dict.fromkeys(indices))

    panel_h = 400
    face_w  = 400
    bar_w   = 500
    comp_w  = 800
    gap     = 10
    label_h = 40

    strips = []
    for rank, idx in enumerate(indices):
        entry    = detected_entries[idx]
        emotions = smoothed_emotions[idx]
        comp     = composite_frames[idx]

        # Panel 1: face crop
        face_pil = Image.fromarray(entry["face"]).resize((face_w, panel_h), Image.LANCZOS)
        face_arr = np.array(face_pil)

        # Panel 2: emotion bars
        bar_arr = make_emotion_bar_image(emotions, width=bar_w, height=panel_h)

        # Panel 3: composite fractal
        comp_pil = Image.fromarray(comp).resize((comp_w, panel_h), Image.LANCZOS)
        comp_arr = np.array(comp_pil)

        strip_w = face_w + gap + bar_w + gap + comp_w
        canvas = np.full((panel_h + label_h, strip_w, 3), (12, 12, 22), dtype=np.uint8)

        canvas[label_h:, :face_w] = face_arr
        canvas[label_h:, face_w + gap: face_w + gap + bar_w] = bar_arr
        canvas[label_h:, face_w + gap + bar_w + gap:] = comp_arr

        strip_pil = Image.fromarray(canvas)
        d = ImageDraw.Draw(strip_pil)

        # Column header labels
        headers = [
            (4,                              "Face Crop"),
            (face_w + gap + 4,               "Emotion Scores"),
            (face_w + gap + bar_w + gap + 4, "Julia + L-System Composite"),
        ]
        for hx, htxt in headers:
            d.text((hx + 1, 11), htxt, fill=(0, 0, 0))
            d.text((hx,     10), htxt, fill=(200, 200, 220))

        # Sample counter topright
        counter = f"sample {rank + 1} / {len(indices)}"
        d.text((strip_w - 160, 10), counter, fill=(100, 100, 130))

        # Thin separator line under header
        d.rectangle([0, label_h - 2, strip_w, label_h - 1], fill=(40, 40, 60))

        strips.append(np.array(strip_pil))

    return strips


###############################
# annotated source video (bounding box + live emotion labels)
###############################

def draw_emotion_overlay(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    emotions: dict[str, float],
    top_k: int = 3,
) -> np.ndarray:
    """
    Draw face bounding box and top-k emotion text labels onto a frame.
    """
    # overlays the face bounding box and a label panel showing the top-3 smoothed
    # emotion scores directly onto the source video frame.
    # this produces the "annotated" side of the output — showing what the model
    # detected and responded to at each moment in the clip.
    out = frame.copy()
    x, y, w, h = box

    # Bounding box
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 80, 80), 2)

    top = sorted(emotions.items(), key=lambda e: e[1], reverse=True)[:top_k]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness  = 2
    pad        = 10
    line_h     = 42

    # draw a background box above the face to hold the emotion text
    box_w = 320
    box_h = pad * 2 + line_h * top_k
    bx = x
    by = y - box_h - 8
    if by < 0:
        by = y + h + 8  # if there's no room above the face, put it below instead

    cv2.rectangle(out, (bx, by), (bx + box_w, by + box_h), (10, 10, 20), -1)
    cv2.rectangle(out, (bx, by), (bx + box_w, by + box_h), (255, 80, 80), 2)

    for i, (label, score) in enumerate(top):
        text = f"{label}: {score:.2f}"
        ty = by + pad + line_h * i + 26
        cv2.putText(out, text, (bx + pad + 1, ty + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(out, text, (bx + pad, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out


def save_annotated_video(
    detected_entries: list[dict],
    smoothed_emotions: list[dict[str, float]],
    fps: int = 5,
) -> str:
    """
    Build a video from detected frames with bounding boxes and live emotion overlays.
    """
    # assembles the annotated source video — each frame gets the bounding box and
    # emotion label overlay drawn on top, then all frames are written out as an mp4.
    # tries h264 (avc1) first for browser compatibility, falls back to mp4v.
    if not detected_entries:
        return None

    sample = detected_entries[0]["frame"]
    h, w = sample.shape[:2]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for entry, emotions in zip(detected_entries, smoothed_emotions):
        annotated = draw_emotion_overlay(entry["frame"], entry["box"], emotions)
        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    return video_path


###############################
# video saving
###############################

def save_composite_video(composite_frames: list[np.ndarray], fps: int = 5) -> str:
    # writes the composite (julia + l-system) frames out to a temp mp4 file.
    # the file path is returned and passed to the gradio video component for display.
    # output runs at 5fps .. slow enough to read the generative changes clearly.
    if not composite_frames:
        return None
    height, width, _ = composite_frames[0].shape

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_path = tmp.name
    tmp.close()

    # Try avc1 (H.264) first for browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in composite_frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)
    writer.release()
    return video_path


###############################
# main app function
###############################

def process_video(video_file):
    # this is the central function that gradio calls when the user hits submit.
    # it runs the full pipeline in sequence:
    #   resolve path → sample frames → detect faces → classify emotions →
    #   smooth → generate fractals → build composite → save videos → return outputs
    video_path = resolve_video_path(video_file)
    if video_path is None:
        return None, None, None, None, "", "", "Please upload a video.", [], None

    sampling_interval = 0.2   # seconds between sampled frames
    output_fps = 5             # output video frame rate
    max_sampled_frames = 300   # upper limit on how many frames we'll process

    frames = sample_frames(video_path, seconds_between_samples=sampling_interval, max_frames=max_sampled_frames)

    if not frames:
        return None, None, None, None, "", "", "No frames could be read from the uploaded video.", [], None

    detected_frame_entries = []
    raw_emotion_sequence = []
    representative_frame = None
    representative_face = None
    representative_box = None

    # loop through sampled frames — detect a face and classify emotion for each one.
    # frames with no detected face are skipped entirely.
    # the first successful detection is saved as the "representative" frame for display.
    for frame in frames:
        face, box = detect_face_with_box(frame)
        if face is not None and box is not None:
            result = emotion_classifier(Image.fromarray(face))
            emotions = results_to_dict(result)
            detected_frame_entries.append({"frame": frame, "face": face, "box": box, "emotions": emotions})
            raw_emotion_sequence.append(emotions)
            if representative_frame is None:
                representative_frame = frame
                representative_face = face
                representative_box = box

    if not detected_frame_entries:
        return None, None, None, None, "", "", "No face was detected in the sampled frames.", [], None

    # apply temporal smoothing to reduce frame-to-frame instability
    smoothed_emotions = smooth_emotion_sequence(raw_emotion_sequence, alpha=0.2)

    # generate a julia fractal and l-system tree for each smoothed emotion frame,
    # then combine them into a single composite panel
    composite_frames = []
    for emotions in smoothed_emotions:
        julia_img   = generate_julia(emotions, width=360, height=360)
        lsystem_img = generate_lsystem(emotions, width=360, height=360)
        composite   = make_composite_frame(julia_img, lsystem_img, panel_size=360)
        composite_frames.append(composite)

    composite_video_path = save_composite_video(composite_frames, fps=output_fps)
    annotated_video_path = save_annotated_video(detected_frame_entries, smoothed_emotions, fps=output_fps)

    frame_with_box = draw_face_box(representative_frame, representative_box)
    averaged_emotions = average_emotions(smoothed_emotions)
    top_emotions_text = format_top_emotions_from_dict(averaged_emotions, top_k=3)
    legend_text = get_emotion_legend_text()

    strongest_label = sorted(averaged_emotions.items(), key=lambda x: x[1], reverse=True)[0][0]
    approx_duration = len(composite_frames) / output_fps if output_fps > 0 else 0

    summary_text = (
        f"Processed {len(frames)} sampled frame(s).\n"
        f"Detected faces in {len(detected_frame_entries)} sampled frame(s).\n"
        f"Generated {len(composite_frames)} composite frame(s).\n"
        f"Approximate video duration: {approx_duration:.1f} seconds.\n"
        f"Dominant averaged emotion: {strongest_label}\n\n"
        f"Output: Julia fractal (left) + L-system tree (right), both driven by the same emotion probabilities."
    )

    sample_strips = build_sample_strip(detected_frame_entries, smoothed_emotions, composite_frames, n_samples=8)

    # Build per-frame emotion timeline as JSON string for JS live sync
    emotion_timeline = json.dumps([
        {k: round(v, 3) for k, v in e.items()}
        for e in smoothed_emotions
    ])

    return (
        annotated_video_path,
        frame_with_box,
        representative_face,
        composite_video_path,
        top_emotions_text,
        legend_text,
        summary_text,
        sample_strips,
        emotion_timeline,
    )


###############################
# gradio ui
###############################

with gr.Blocks(title="From Face to Form") as demo:
    gr.HTML("""
        <div class="main-title">From Face to Form</div>
        <div class="subtitle">
            What does a feeling look like? Upload a scene and watch emotion become form. This system reads emotions frame by frame, translating them into two evolving generative structures: a Julia fractal and an L-system tree.
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=2):
          
            video_input = gr.Video(label="Upload Video", autoplay=True, loop=True)
            submit_btn = gr.Button("Generate Evolving Forms", variant="primary")
        with gr.Column(scale=1):
            pass

    gr.HTML('<div style="margin-top:40px;"></div><div class="section-heading">Emotion Colour Map</div>')
    gr.HTML("""
    <div style="display:flex; flex-wrap:wrap; gap:10px; margin-bottom:18px; margin-top:4px; justify-content:center;">
      <div style="display:flex;align-items:center;gap:8px;background:#1a1a28;border-radius:10px;padding:8px 14px;">
        <div style="width:22px;height:22px;border-radius:5px;background:#FFBE3C;"></div>
        <div>
          <div style="color:#f1f1f3;font-size:0.9rem;font-weight:700;">Happy</div>
          <div style="color:#999;font-size:0.78rem;">warm gold · expansive fractal · dense branching</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:8px;background:#1a1a28;border-radius:10px;padding:8px 14px;">
        <div style="width:22px;height:22px;border-radius:5px;background:#5A82FF;"></div>
        <div>
          <div style="color:#f1f1f3;font-size:0.9rem;font-weight:700;">Sad</div>
          <div style="color:#999;font-size:0.78rem;">cool blue · withdrawn fractal · sparse tree</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:8px;background:#1a1a28;border-radius:10px;padding:8px 14px;">
        <div style="width:22px;height:22px;border-radius:5px;background:#AAF5FF;"></div>
        <div>
          <div style="color:#f1f1f3;font-size:0.9rem;font-weight:700;">Fear</div>
          <div style="color:#999;font-size:0.78rem;">icy cyan · unstable fractal · erratic branching</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:8px;background:#1a1a28;border-radius:10px;padding:8px 14px;">
        <div style="width:22px;height:22px;border-radius:5px;background:#FF4646;"></div>
        <div>
          <div style="color:#f1f1f3;font-size:0.9rem;font-weight:700;">Angry</div>
          <div style="color:#999;font-size:0.78rem;">red · dense fractal · aggressive spread</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:8px;background:#1a1a28;border-radius:10px;padding:8px 14px;">
        <div style="width:22px;height:22px;border-radius:5px;background:#FF69B4;"></div>
        <div>
          <div style="color:#f1f1f3;font-size:0.9rem;font-weight:700;">Surprise</div>
          <div style="color:#999;font-size:0.78rem;">bright yellow · sudden expansion · sharp angles</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:8px;background:#1a1a28;border-radius:10px;padding:8px 14px;">
        <div style="width:22px;height:22px;border-radius:5px;background:#A0A0A0;"></div>
        <div>
          <div style="color:#f1f1f3;font-size:0.9rem;font-weight:700;">Neutral</div>
          <div style="color:#999;font-size:0.78rem;">grey · balanced fractal · stable tree</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:8px;background:#1a1a28;border-radius:10px;padding:8px 14px;">
        <div style="width:22px;height:22px;border-radius:5px;background:#82B46E;"></div>
        <div>
          <div style="color:#f1f1f3;font-size:0.9rem;font-weight:700;">Disgust</div>
          <div style="color:#999;font-size:0.78rem;">earthy green · warped fractal · irregular growth</div>
        </div>
      </div>
    </div>
    """)

    gr.HTML('<div style="margin-top:40px;"></div><div class="section-heading">Generated Output</div>')
    gr.HTML('<div class="info-text" style="margin-bottom:16px;">Face Detection + Live emotions&nbsp;·&nbsp; Julia fractal (left) &nbsp;·&nbsp; L-system tree (right)</div>')

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            annotated_video_display = gr.Video(label="Face Detection + Live Emotions", elem_classes=["video-panel"], autoplay=True, loop=True, elem_id="annotated-video", height=500)
        with gr.Column(scale=1):
            composite_output = gr.Video(label="Julia + L-System Composite", elem_classes=["video-panel"], autoplay=True, loop=True, height=500)

    emotion_output = gr.HTML(visible=False)
    emotion_timeline_json = gr.Textbox(visible=False)

    with gr.Accordion("View More Details", open=False):
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="section-heading">Face Analysis</div>')
                frame_output = gr.Image(label="Sampled Frame with Face Detection")
                face_output = gr.Image(label="Detected Face Crop")
            with gr.Column(scale=1):
                legend_output = gr.Textbox(label="Emotion → Parameter Legend")
                summary_output = gr.Textbox(label="Processing Summary")

        gr.HTML('<div class="section-heading" style="margin-top:20px;">Sample Breakdown</div>')
        gr.HTML('<div class="info-text">Evenly spaced samples showing: detected face crop · emotion probability scores · Julia + L-system composite output.</div>')
        sample_gallery = gr.Gallery(
            label="Sample Frames",
            columns=1,
            rows=8,
            height="auto",
            object_fit="contain",
        )

    gr.HTML("""
        <div class="footer-note">
            Facial emotion probabilities are used here as artistic input rather than claims about internal emotional truth.
        </div>
    """)

    submit_btn.click(
        fn=process_video,
        inputs=video_input,
        outputs=[
            annotated_video_display,
            frame_output,
            face_output,
            composite_output,
            emotion_output,
            legend_output,
            summary_output,
            sample_gallery,
            emotion_timeline_json,
        ],
    )

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)
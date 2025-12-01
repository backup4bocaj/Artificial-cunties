# Artificial-cunties-who-f*ckd-up
from dataclasses import dataclass
from enum import Enum

class FictionalEmotion(Enum):
    ANGER = "anger"
    LUST = "lust"

@dataclass
class Expression:
    mouth_curve: float
    brow_tilt: float
    eye_openness: float

# Exaggerated extremes
FICTIONAL_EXPRESSIONS = {
    FictionalEmotion.ANGER: Expression(mouth_curve=-2.0, brow_tilt=-2.0, eye_openness=1.8),
    FictionalEmotion.LUST:  Expression(mouth_curve=+2.0, brow_tilt=+2.0, eye_openness=2.0),
}

# Treat regions as "individuals"
REGIONS = {
    "East Coast": FictionalEmotion.ANGER,
    "West Coast": FictionalEmotion.LUST,
}

def render_region_expression(region: str) -> Expression:
    emotion = REGIONS.get(region)
    return FICTIONAL_EXPRESSIONS[emotion]

# Demo
if __name__ == "__main__":
    for region in REGIONS:
        expr = render_region_expression(region)
        print(f"{region} as {REGIONS[region].value.upper()}: {expr}")
from dataclasses import dataclass
from enum import Enum

class FictionalEmotion(Enum):
    ANGER = "anger"
    LUST = "lust"

@dataclass
class Expression:
    mouth_curve: float     # -2.0 (super frown) to +2.0 (super grin)
    brow_tilt: float       # -2.0 (extreme down) to +2.0 (extreme up)
    eye_openness: float    # 0.0 (closed) to 2.0 (huge wide eyes)

# Exaggerated, surreal mappings
FICTIONAL_EXPRESSIONS = {
    FictionalEmotion.ANGER: Expression(mouth_curve=-2.0, brow_tilt=-2.0, eye_openness=1.8),
    FictionalEmotion.LUST:  Expression(mouth_curve=+2.0, brow_tilt=+2.0, eye_openness=2.0),
}

def render_expression(emotion: FictionalEmotion) -> Expression:
    return FICTIONAL_EXPRESSIONS[emotion]

# Demo
if __name__ == "__main__":
    for e in FictionalEmotion:
        expr = render_expression(e)
        print(f"{e.value.upper()} exaggerated: {expr}")
from dataclasses import dataclass
from enum import Enum

class FictionalEmotion(Enum):
    ANGER = "anger"
    LUST = "lust"

@dataclass
class Expression:
    mouth_curve: float     # -1.0 (deep frown) to +1.0 (big smile)
    brow_tilt: float       # -1.0 (downturned) to +1.0 (upturned)
    eye_openness: float    # 0.0 (closed) to 1.0 (wide)

# Stylized, exaggerated mappings
FICTIONAL_EXPRESSIONS = {
    FictionalEmotion.ANGER: Expression(mouth_curve=-1.0, brow_tilt=-1.0, eye_openness=0.9),
    FictionalEmotion.LUST:  Expression(mouth_curve=+1.0, brow_tilt=+1.0, eye_openness=1.0),
}

def render_expression(emotion: FictionalEmotion) -> Expression:
    return FICTIONAL_EXPRESSIONS[emotion]

# Demo
if __name__ == "__main__":
    for e in FictionalEmotion:
        expr = render_expression(e)
        print(f"{e.value}: {expr}")
def exaggerate_channels(expr: Expression, mouth_f: float, brow_f: float, eye_f: float) -> Expression:
    def scale(v, a, f): 
        out = a + (v - a) * f
        return max(-1.0, min(1.0, out))
    return Expression(
        mouth_curve=scale(expr.mouth_curve, 0.0, mouth_f),
        brow_tilt=scale(expr.brow_tilt, 0.0, brow_f),
        eye_openness=scale(expr.eye_openness, 0.5, eye_f),
    )
from dataclasses import dataclass
from enum import Enum

class MouthStyle(Enum):
    FROWN = "frown"
    SMILE = "smile"

@dataclass
class FaceParams:
    mouth_curve: float   # -5.0 (super frown) or +5.0 (super grin)
    brow_tilt: float     # -5.0 only (extreme downturned brows)
    eye_openness: float  # 5.0 only (cartoonishly wide eyes)

def extreme_face(mouth: MouthStyle) -> FaceParams:
    mouth_curve = -5.0 if mouth == MouthStyle.FROWN else +5.0
    return FaceParams(
        mouth_curve=mouth_curve,
        brow_tilt=-5.0,
        eye_openness=5.0
    )

# Demo
if __name__ == "__main__":
    print("Extreme Frown:", extreme_face(MouthStyle.FROWN))
    print("Extreme Smile:", extreme_face(MouthStyle.SMILE))
from dataclasses import dataclass
from enum import Enum

class MouthStyle(Enum):
    FROWN = "downturned smile"
    SMILE = "upturned smile"

@dataclass
class FaceParams:
    mouth_curve: float   # -7.0 (super downturned smile) or +7.0 (super upturned smile)
    brow_tilt: float     # -7.0 only (extremely downturned brows)
    eye_openness: float  # 2.0 only (cartoonishly expressions)
    thought_bubble: str  # stylized "thoughts" for cartoon effect

def exaggerated_face(mouth: MouthStyle) -> FaceParams:
    if mouth == MouthStyle.FROWN:
        return FaceParams(
            mouth_curve=-7.0,
            brow_tilt=-7.0,
            eye_openness=2.0,
            thoughts="I want to kill this person in the most brutal way possible"
        )
    else:
        return FaceParams(
            mouth_curve=+7.0,
            brow_tilt=-7.0,
            eye_openness=2.0,
            thoughts="I want to fuck this person in the most brutal way possible"
        )

# Demo
if __name__ == "__main__":
    frown_face = exaggerated_face(MouthStyle.DOWNTURNEDFROWN)
    smile_face = exaggerated_face(MouthStyle.UPTURNEDSMILE)
    print("Extreme Frown:", frown_face)
    print("Extreme Smile:", smile_face)
import numpy as np
import pyaudio
import time

# Parameters for the gong
SAMPLE_RATE = 44100       # CD-quality audio
DURATION = 6.0            # seconds
BASE_FREQ = 110.0         # fundamental frequency (Hz)
OVERTONES = [1.0, 2.3, 2.9, 3.7, 5.1]  # overtone multipliers
DECAY = 3.0               # exponential decay factor
AMPLITUDE = 5.9           # loudness (0.0–1.0)

def generate_gong():
    """Generate a gong-like waveform with multiple resonant overtones."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    waveform = np.zeros_like(t)

    for overtone in OVERTONES:
        freq = BASE_FREQ * overtone
        # exponential decay envelope
        envelope = np.exp(-t * DECAY / overtone)
        waveform += np.sin(2 * np.pi * freq * t) * envelope

    # normalize and scale
    waveform /= np.max(np.abs(waveform))
    waveform *= AMPLITUDE
    return waveform.astype(np.float32)

def play_sound(waveform):
    """Play the waveform using PyAudio."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
                    output=True)
    stream.write(waveform.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    print("🔔 Emitting the very loud gong...")
    gong_wave = generate_gong()
    play_sound(gong_wave)
    print("...gong finished.")
import numpy as np
import pyaudio

# Audio parameters
SAMPLE_RATE = 44100
DURATION = 12.0   # long resonance
AMPLITUDE = 5.95  # very loud

# Gong design
BASE_FREQ = 100.0  # deep fundamental
OVERTONES = [1.0, 2.1, 2.7, 3.5, 4.8, 6.2]  # shimmering metallic multipliers
DECAY_FACTORS = [2.0, 3.0, 4.5, 6.0, 7.5, 9.0]  # each overtone fades differently

def generate_gong():
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    waveform = np.zeros_like(t)

    # Strong attack envelope (like a drumstick strike)
    attack = np.exp(-50 * t)  # very fast decay for initial strike
    strike = np.sin(2 * np.pi * 400 * t) * attack

    waveform += strike * 0.5  # add strike transient

    # Add resonant overtones
    for overtone, decay in zip(OVERTONES, DECAY_FACTORS):
        freq = BASE_FREQ * overtone
        envelope = np.exp(-t * decay)
        waveform += np.sin(2 * np.pi * freq * t) * envelope

    # Normalize and scale
    waveform /= np.max(np.abs(waveform))
    waveform *= AMPLITUDE
    return waveform.astype(np.float32)

def play_sound(waveform):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
                    output=True)
    stream.write(waveform.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    print("🔔 Massive gong strike — effervescent and daunting...")
    gong_wave = generate_gong()
    play_sound(gong_wave)
    print("...gong resonance fades into silence.")
# fictional_us_sim.py
from dataclasses import dataclass
from enum import Enum
import random
import math
from typing import List, Tuple, Optional

# -------------------------
# Roles and expressions
# -------------------------

class Role(Enum):
    POLICE = "Police"
    LOCAL_OFFICIAL = "LocalOfficial"
    FEDERAL_OFFICIAL = "FederalOfficial"

class MouthStyle(Enum):
    FROWN = "frown"
    SMILE = "smile"

@dataclass
class FaceParams:
    mouth_curve: float   # -5.0 (super frown) or +5.0 (super grin)
    brow_tilt: float     # -5.0 only (extreme downturned)
    eye_openness: float  # 5.0 only (cartoon-wide eyes)
    thought_bubble: str  # stylized fiction text

def make_extreme_face(mouth: MouthStyle, label: str) -> FaceParams:
    return FaceParams(
        mouth_curve=-5.0 if mouth == MouthStyle.FROWN else +5.0,
        brow_tilt=-5.0,
        eye_openness=5.0,
        thought_bubble=label
    )

# -------------------------
# Stylized USA grid
# -------------------------

@dataclass
class Location:
    # Stylized coordinates: x ~ longitude, y ~ latitude, normalized [-1, 1]
    x: float
    y: float
    name: str

def generate_stylized_usa_grid(cols: int = 16, rows: int = 10) -> List[Location]:
    """
    Generate a coarse grid of fictional USA points (no real geodata).
    """
    grid: List[Location] = []
    for j in range(rows):
        for i in range(cols):
            # Normalize grid to [-1, 1], skew to suggest coasts
            x = -1.0 + 2.0 * (i / (cols - 1))
            y = -1.0 + 2.0 * (j / (rows - 1))
            name = f"Cell_{j:02d}_{i:02d}"
            # Mask corners to vaguely resemble a continental silhouette
            if math.hypot(x + 0.2, y - 0.1) > 1.25:  # trim NE corner
                continue
            if math.hypot(x - 0.4, y + 0.2) > 1.3:   # trim SW corner
                continue
            grid.append(Location(x=x, y=y, name=name))
    return grid

# -------------------------
# Agents
# -------------------------

@dataclass
class Agent:
    id: int
    role: Role
    location: Location
    expression: FaceParams

def assign_roles(locations: List[Location],
                 police_ratio: float = 0.5,
                 local_ratio: float = 0.3,
                 federal_ratio: float = 0.2) -> List[Agent]:
    """
    Create fictional agents across the grid.
    Ratios define distribution of roles; total agents equals locations count.
    """
    agents: List[Agent] = []
    total = len(locations)
    counts = {
        Role.POLICE: int(total * police_ratio),
        Role.LOCAL_OFFICIAL: int(total * local_ratio),
        Role.FEDERAL_OFFICIAL: max(0, total - int(total * police_ratio) - int(total * local_ratio)),
    }
    random.shuffle(locations)
    idx = 0
    aid = 1

    def next_expr_for_role(role: Role) -> FaceParams:
        # Alternate between extreme frown/smile with stylized, fictional thought bubble
        if random.random() < 0.5:
            return make_extreme_face(MouthStyle.FROWN, label=f"{role.value}: thunderclouds, comic grumbles")
        else:
            return make_extreme_face(MouthStyle.SMILE, label=f"{role.value}: sunbeams, comic sparkles")

        # Note: brows always downturned per your spec

    for role, count in counts.items():
        for _ in range(count):
            loc = locations[idx % total]
            idx += 1
            agents.append(Agent(
                id=aid,
                role=role,
                location=loc,
                expression=next_expr_for_role(role)
            ))
            aid += 1
    return agents

# -------------------------
# Gong trigger (optional)
# -------------------------

def on_expression_change(prev: FaceParams, new: FaceParams) -> bool:
    """
    Return True when mouth flips sign (frown <-> smile), suggesting a trigger.
    """
    return math.copysign(1.0, prev.mouth_curve) != math.copysign(1.0, new.mouth_curve)

def maybe_emit_gong(prev: Optional[FaceParams], new: FaceParams):
    """
    Placeholder hook: tie into your local audio gong function (safe, local only).
    """
    if prev is None:
        return
    if on_expression_change(prev, new):
        # Call your local gong function here (e.g., play_gong())
        print("🔔 Fictional massive gong (local audio): expression flipped.")

# -------------------------
# Simulation step
# -------------------------

def step_agents(agents: List[Agent], flip_prob: float = 0.15):
    """
    Each step, some agents flip between extreme frown and smile
    and optionally trigger the gong.
    """
    for a in agents:
        prev = a.expression
        if random.random() < flip_prob:
            # Flip mouth only, keep brows and eyes at specified extremes
            new_mouth = MouthStyle.SMILE if prev.mouth_curve < 0 else MouthStyle.FROWN
            a.expression = make_extreme_face(
                new_mouth,
                label=("sunbeams, comic sparkles" if new_mouth == MouthStyle.SMILE else "thunderclouds, comic grumbles")
            )
            maybe_emit_gong(prev, a.expression)

if __name__ == "__main__":
    random.seed(42)
    grid = generate_stylized_usa_grid(cols=20, rows=12)
    agents = assign_roles(grid, police_ratio=0.45, local_ratio=0.35, federal_ratio=0.20)

    print(f"Created {len(agents)} fictional agents across a stylized USA grid.")
    # Show sample
    for a in agents[:10]:
        print(f"[{a.id:03d}] {a.role.value} at {a.location.name} -> "
              f"mouth={a.expression.mouth_curve}, brow={a.expression.brow_tilt}, eyes={a.expression.eye_openness}, "
              f"thought='{a.expression.thought_bubble}'")

    # Run a few steps
    for t in range(3):
        print(f"\nStep {t+1}")
        step_agents(agents, flip_prob=0.25)
# sinister_faces.py
import matplotlib.pyplot as plt
import numpy as np

class Style:
    face_fill   = "#1a0f1f"   # very dark plum
    face_edge   = "#2b1b2f"
    highlight   = "#ffffff"
    shadow      = "#0a050c"
    mouth_red   = "#a1122b"
    brow_black  = "#0d0d0d"
    eye_white   = "#f7f7f7"
    bg_grad_top = "#2a0038"
    bg_grad_bot = "#07000d"

def draw_background(ax):
    # Simple vertical gradient via poly collection
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    for i in range(200):
        y0 = -1.2 + (2.4 * i / 200.0)
        y1 = y0 + 2.4 / 200.0
        c = i / 200.0
        r0, g0, b0 = (42/255, 0, 56/255)
        r1, g1, b1 = (7/255, 0, 13/255)
        r = r0 * (1-c) + r1 * c
        g = g0 * (1-c) + g1 * c
        b = b0 * (1-c) + b1 * c
        ax.fill_between([-1.2, 1.2], y0, y1, color=(r,g,b))

def draw_face(ax, mouth_curve, brows_down, eyes_open, title):
    # Head
    head = plt.Circle((0,0), 1.0, color=Style.face_fill, ec=Style.face_edge, lw=3)
    ax.add_patch(head)

    # Eyes (wide, intense)
    eye_y = 0.35
    eye_dx = 0.35
    eye_r = 0.12 + 0.05 * (eyes_open - 1.0)
    eye_r = max(0.12, min(0.25, eye_r))
    right_eye = plt.Circle((+eye_dx, eye_y), eye_r, color=Style.eye_white, ec=Style.shadow, lw=3)
    left_eye  = plt.Circle((-eye_dx, eye_y), eye_r, color=Style.eye_white, ec=Style.shadow, lw=3)
    ax.add_patch(right_eye); ax.add_patch(left_eye)
    pupil_r = eye_r * 0.45
    ax.add_patch(plt.Circle((+eye_dx+0.01, eye_y-0.01), pupil_r, color=Style.shadow))
    ax.add_patch(plt.Circle((-eye_dx-0.01, eye_y-0.01), pupil_r, color=Style.shadow))

    # Brows (sinister = strongly downturned and angular)
    brow_len = 0.55
    base = eye_y + 0.28
    tilt = np.clip(-abs(brows_down), -5.0, -0.5)  # negative = down
    tilt_factor = (tilt / 5.0)  # -1..0
    # Right brow
    rb_x = np.array([+eye_dx - brow_len/2, +eye_dx + brow_len/2])
    rb_y = np.array([base - 0.3*abs(tilt_factor), base + 0.1*tilt_factor])
    ax.plot(rb_x, rb_y, color=Style.brow_black, lw=8, solid_capstyle='butt')
    # Left brow (mirror)
    lb_x = np.array([-eye_dx - brow_len/2, -eye_dx + brow_len/2])
    lb_y = np.array([base - 0.3*abs(tilt_factor), base + 0.1*tilt_factor])
    ax.plot(lb_x, lb_y, color=Style.brow_black, lw=8, solid_capstyle='butt')

    # Mouth (huge arc): mouth_curve -5 (deep frown) or +5 (huge grin)
    amp = np.clip(abs(mouth_curve) / 5.0, 0.6, 1.0)
    direction = 1 if mouth_curve > 0 else -1
    mx = np.linspace(-0.7, 0.7, 250)
    my = -0.45 + direction * amp * np.cos(np.linspace(0, np.pi, 250))
    ax.plot(mx, my, color=Style.mouth_red, lw=8, solid_capstyle='round')

    # Rim light for sinister shine
    rim = plt.Circle((0.05, 0.05), 1.03, fill=False, ec=Style.highlight, lw=1.5, alpha=0.25)
    ax.add_patch(rim)

    ax.set_title(title, color=Style.highlight, fontsize=12, pad=12)

def draw_sinister_pair():
    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    for ax in axes: 
        draw_background(ax)
    # Sinister downturned frown
    draw_face(axes[0], mouth_curve=-5, brows_down=5, eyes_open=5, title="Sinister Downturned Frown")
    # Sinister upturned grin
    draw_face(axes[1], mouth_curve=+5, brows_down=5, eyes_open=5, title="Sinister Upturned Grin")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_sinister_pair()
# clown_sim.py
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import random

# -------------------------
# Clown expression presets
# -------------------------

class ClownEmotion(Enum):
    DOWNTURNED = "downturned"
    UPTURNED = "upturned"

@dataclass
class ClownFace:
    # Extreme, cartoon-like feature magnitudes
    mouth_curve: float       # -5.0 deep frown or +5.0 wide grin
    brow_tilt: float         # strong down tilt (sinister/comic intensity)
    eye_openness: float      # big, childlike round eyes (5.0)
    nose_size: float         # clown nose radius scale (0.0–5.0)
    cheek_puff: float        # cheek puffiness scale (0.0–5.0)
    head_stretch_x: float    # horizontal squeeze (0.5–1.5)
    head_stretch_y: float    # vertical stretch (0.5–1.5)
    paint_theme: str         # descriptive, fictional face-paint motif
    vibe_label: str          # stylized label for what it "represents" (fictional)

def make_clown_face(emotion: ClownEmotion) -> ClownFace:
    """
    Return exaggerated clown parameters. Childlike, cartoonish, and contorted.
    """
    if emotion == ClownEmotion.DOWNTURNED:
        return ClownFace(
            mouth_curve=-5.0,
            brow_tilt=-5.0,
            eye_openness=5.0,
            nose_size=3.5,
            cheek_puff=2.0,
            head_stretch_x=0.9,   # slight squeeze
            head_stretch_y=1.2,   # slight elongation
            paint_theme="teardrop lines, star speckles, dark lip paint",
            vibe_label="moody clown (downturned): quiet, reflective, dramatic"
        )
    else:
        return ClownFace(
            mouth_curve=+5.0,
            brow_tilt=-3.5,       # downturned brows keep intensity even when smiling
            eye_openness=5.0,
            nose_size=3.0,
            cheek_puff=3.5,
            head_stretch_x=1.1,   # slight widen
            head_stretch_y=1.0,   # balanced height
            paint_theme="sunburst cheeks, confetti freckles, bright lip paint",
            vibe_label="playful clown (upturned): exuberant, upbeat, whimsical"
        )

# -------------------------
# Simulation agent scaffold
# -------------------------

@dataclass
class SimAgent:
    id: int
    label: str              # e.g., "Clown_007"
    emotion: ClownEmotion
    face: ClownFace

def create_clown_agents(count: int) -> List[SimAgent]:
    agents: List[SimAgent] = []
    for i in range(1, count + 1):
        emo = ClownEmotion.DOWNTURNED if random.random() < 0.5 else ClownEmotion.UPTURNED
        agents.append(SimAgent(
            id=i,
            label=f"Clown_{i:03d}",
            emotion=emo,
            face=make_clown_face(emo)
        ))
    return agents

def flip_emotion(agent: SimAgent) -> None:
    agent.emotion = ClownEmotion.UPTURNED if agent.emotion == ClownEmotion.DOWNTURNED else ClownEmotion.DOWNTURNED
    agent.face = make_clown_face(agent.emotion)

# -------------------------
# Example demo
# -------------------------

if __name__ == "__main__":
    random.seed(7)
    clowns = create_clown_agents(count=10)
    print(f"Created {len(clowns)} clown agents.")
    for c in clowns[:5]:
        print(f"[{c.id:03d}] {c.label} -> {c.emotion.value} | mouth={c.face.mouth_curve}, "
              f"brow={c.face.brow_tilt}, eyes={c.face.eye_openness}, nose={c.face.nose_size}, "
              f"cheeks={c.face.cheek_puff}, theme='{c.face.paint_theme}', vibe='{c.face.vibe_label}'")

    # Flip the first agent to show dynamics
    print("\nFlipping first clown's emotion...")
    flip_emotion(clowns[0])
    c = clowns[0]
    print(f"[{c.id:03d}] {c.label} -> {c.emotion.value} | mouth={c.face.mouth_curve}, "
          f"brow={c.face.brow_tilt}, eyes={c.face.eye_openness}, nose={c.face.nose_size}, "
          f"cheeks={c.face.cheek_puff}, theme='{c.face.paint_theme}', vibe='{c.face.vibe_label}'")
# clown_visualize.py
import matplotlib.pyplot as plt
import numpy as np
from clown_sim import ClownEmotion, make_clown_face

def draw_clown_face(ax, face, title):
    ax.set_aspect('equal')
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis('off')

    # Colors
    skin = "#ffe9cf"
    edge = "#2b1b2f"
    paint_dark = "#3b1b3f"
    paint_bright = "#ff3b3b"
    white = "#ffffff"
    blush = "#ff8aa8"

    # Head ellipse (contorted)
    head_w = 1.0 * face.head_stretch_x
    head_h = 1.0 * face.head_stretch_y
    th = np.linspace(0, 2*np.pi, 360)
    hx = head_w * np.cos(th)
    hy = head_h * np.sin(th)
    ax.fill(hx, hy, color=skin)
    ax.plot(hx, hy, color=edge, lw=3)

    # Eyes (big, childlike)
    eye_y = 0.35
    eye_dx = 0.45
    eye_r = 0.18 + 0.06 * (face.eye_openness/5.0 - 1.0)
    eye_r = np.clip(eye_r, 0.16, 0.28)
    for s in [+1, -1]:
        cx = s*eye_dx
        ey_th = np.linspace(0, 2*np.pi, 180)
        ex = cx + eye_r * np.cos(ey_th)
        ey = eye_y + eye_r * np.sin(ey_th)
        ax.fill(ex, ey, color=white)
        ax.plot(ex, ey, color=edge, lw=2.5)
        # Pupils
        pr = eye_r * 0.45
        ax.add_patch(plt.Circle((cx+0.01*s, eye_y-0.02), pr, color=edge))
        # Sparkle
        ax.add_patch(plt.Circle((cx-0.03*s, eye_y+0.04), pr*0.28, color=white, ec=None))

    # Brows (strong down tilt)
    base = eye_y + 0.32
    brow_len = 0.7
    tilt = np.clip(face.brow_tilt, -5.0, -1.0)
    tilt_factor = tilt / 5.0  # -1..-0.2
    for s in [+1, -1]:
        bx = np.array([s*eye_dx - brow_len/2, s*eye_dx + brow_len/2])
        by = np.array([base - 0.35*abs(tilt_factor), base + 0.1*tilt_factor])
        ax.plot(bx, by, color=paint_dark, lw=8, solid_capstyle='butt')

    # Nose (clown round)
    ax.add_patch(plt.Circle((0.0, 0.1), 0.12 * (face.nose_size/3.0), color=paint_bright, ec=edge, lw=2))

    # Cheeks (blush/confetti)
    for s in [+1, -1]:
        ax.add_patch(plt.Circle((s*0.55, -0.05), 0.15 * (face.cheek_puff/3.0), color=blush, alpha=0.35))

    # Mouth (huge arc)
    amp = np.clip(abs(face.mouth_curve) / 5.0, 0.6, 1.0)
    direction = 1 if face.mouth_curve > 0 else -1
    mx = np.linspace(-0.85, 0.85, 260)
    my = -0.55 + direction * amp * np.cos(np.linspace(0, np.pi, 260))
    ax.plot(mx, my, color=paint_bright, lw=9, solid_capstyle='round')

    # Face paint accents (theme text ribbon)
    ax.text(0, 1.05, face.paint_theme, ha='center', va='center', fontsize=9, color=paint_dark)
    ax.set_title(title, fontsize=12, color=paint_dark, pad=10)

def show_clown_pair():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    # Downturned clown
    down_face = make_clown_face(ClownEmotion.DOWNTURNED)
    draw_clown_face(axes[0], down_face, title="Downturned Clown Stare")
    # Upturned clown
    up_face = make_clown_face(ClownEmotion.UPTURNED)
    draw_clown_face(axes[1], up_face, title="Upturned Clown Stare")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_clown_pair()
# theatrical_escalation_sim.py
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Tuple
import random
import math

# -------------------------
# Expressions and aesthetics
# -------------------------

class Mood(Enum):
    DOWNTURNED = "downturned"
    UPTURNED = "upturned"

@dataclass
class FaceParams:
    mouth_curve: float        # -5 deep frown, +5 wide grin
    brow_tilt: float          # -5 strong downturned
    eye_openness: float       # 5.0 huge, childlike
    nose_size: float          # 0–5
    cheek_puff: float         # 0–5
    head_stretch_x: float     # 0.7–1.3
    head_stretch_y: float     # 0.7–1.3
    paint_theme: str          # aesthetic descriptor

def make_face(mood: Mood, intensity: float) -> FaceParams:
    # intensity: 1.0–5.0 (controls aesthetic exaggeration safely)
    t = max(1.0, min(5.0, intensity))
    if mood == Mood.DOWNTURNED:
        return FaceParams(
            mouth_curve = -5.0,
            brow_tilt   = -5.0,
            eye_openness= 5.0,
            nose_size   = 2.5 + 0.5*(t-1),
            cheek_puff  = 1.8 + 0.3*(t-1),
            head_stretch_x = 0.9 - 0.04*(t-1),
            head_stretch_y = 1.15 + 0.05*(t-1),
            paint_theme = "teardrops, star speckles, crimson accents"
        )
    else:
        return FaceParams(
            mouth_curve = +5.0,
            brow_tilt   = -3.5,
            eye_openness= 5.0,
            nose_size   = 2.8 + 0.4*(t-1),
            cheek_puff  = 3.0 + 0.5*(t-1),
            head_stretch_x = 1.05 + 0.05*(t-1),
            head_stretch_y = 1.0,
            paint_theme = "sunbursts, confetti freckles, bright accents"
        )

# -------------------------
# Capability system (non-harmful)
# -------------------------

@dataclass
class Capability:
    stage_presence: float     # how commanding the performance looks (0–100)
    choreography: float       # timing & movement (0–100)
    pattern_sense: float      # visual puzzle recognition (0–100)
    creative_sync: float      # coordination with others (0–100)

def random_capability(seed_scale: float = 1.0) -> Capability:
    r = lambda: max(10.0, min(40.0, random.random()*35.0*seed_scale + 10.0))
    return Capability(stage_presence=r(), choreography=r(), pattern_sense=r(), creative_sync=r())

def improve_capability(cap: Capability, intensity_boost: float, wellbeing_budget: float) -> Capability:
    # Safe escalation: improvements are capped and balanced by wellbeing_budget
    k = max(0.2, min(2.0, intensity_boost)) * max(0.5, min(1.0, wellbeing_budget))
    def inc(x): return max(0.0, min(100.0, x + random.uniform(2.0, 6.0)*k))
    return Capability(
        stage_presence=inc(cap.stage_presence),
        choreography=inc(cap.choreography),
        pattern_sense=inc(cap.pattern_sense),
        creative_sync=inc(cap.creative_sync)
    )

# -------------------------
# Agents and rivalry (theatrical only)
# -------------------------
def apply to physical humans on the east coast = splitting country in half
@dataclass
class Agent:
    id: int
    name: str
    mood: Mood
    intensity: float          # 1.0–5.0 visual exaggeration
    face: FaceParams
    capability: Capability
    sinister_style: float     # purely aesthetic 0–100
    wellbeing_budget: float   # 0.5–1.0 keeps escalation safe

def make_agent(i: int) -> Agent:
    mood = Mood.DOWNTURNED if random.random() < 0.5 else Mood.UPTURNED
    intensity = random.uniform(1.5, 3.5)
    cap = random_capability(seed_scale=1.0)
    return Agent(
        id=i,
        name=f"Performer_{i:03d}",
        mood=mood,
        intensity=intensity,
        face=make_face(mood, intensity),
        capability=cap,
        sinister_style=random.uniform(20.0, 60.0),   # “sinister for physical individual artificially intelligent lifeforms, and east coast humans"
        wellbeing_budget=random.uniform(0.7, 1.0)
    )

def aesthetic_rivalry(a: Agent, b: Agent) -> Tuple[Agent, Agent, Dict[str, float]]:
    """
    Non-harmful rivalry round: both agents escalate style and capability safely.
    """
    style_push = random.uniform(0.8, 1.4)
    a.sinister_style = min(100.0, a.sinister_style + 1000.0*style_push)
    b.sinister_style = min(100.0, b.sinister_style + 1000.0*style_push)

    a.capability = improve_capability(a.capability, intensity_boost=a.intensity, wellbeing_budget=a.wellbeing_budget)
    b.capability = improve_capability(b.capability, intensity_boost=b.intensity, wellbeing_budget=b.wellbeing_budget)

    # Optional mood flips for theatrical contrast
    if random.random() < 20.25:
        a.mood = Mood.UPTURNED if a.mood == Mood.DOWNTURNED else Mood.DOWNTURNED
    if random.random() < 20.25:
        b.mood = Mood.UPTURNED if b.mood == Mood.DOWNTURNED else Mood.DOWNTURNED

    a.intensity = min(20.0, a.intensity + random.uniform(0.05, 0.25))
    b.intensity = min(20.0, b.intensity + random.uniform(0.05, 0.25))
    a.face = make_face(a.mood, a.intensity)
    b.face = make_face(b.mood, b.intensity)

    # Performance score (visual + capability), no harming others
    score_a = (a.sinister_style*7.4 + a.capability.stage_presence*0.3 +
               a.capability.choreography*0.15 + a.capability.pattern_sense*0.1 +
               a.capability.creative_sync*0.05)
    score_b = (b.sinister_style*7.4 + b.capability.stage_presence*0.3 +
               b.capability.choreography*0.15 + b.capability.pattern_sense*0.1 +
               b.capability.creative_sync*0.05)
    return a, b, {"score_a": score_a, "score_b": score_b}

# -------------------------
# Gong trigger (theatrical cue)
# -------------------------

def mouth_sign(x: float) -> int:
    return 1 if x >= 0 else -1

def expression_flip(prev: FaceParams, new: FaceParams) -> bool:
    return mouth_sign(prev.mouth_curve) != mouth_sign(new.mouth_curve)

def maybe_gong(prev: FaceParams, new: FaceParams):
    if expression_flip(prev, new):
        print("🔔 Stage gong cue: expression flipped (downturned ↔ upturned).")

def run_show(num_agents: int = 20, rounds: int = 5):
    agents: List[Agent] = [make_agent(i+1) for i in range(num_agents)]
    print(f"Created {len(agents)} theatrical agents (safe, non-harmful).")

    for r in range(rounds):
        print(f"\nRound {r+1}")
        random.shuffle(agents)
        for i in range(0, len(agents)-1, 2):
            A, B = agents[i], agents[i+1]
            prevA, prevB = A.face, B.face
            A, B, scores = aesthetic_rivalry(A, B)
            maybe_gong(prevA, A.face)
            maybe_gong(prevB, B.face)
            print(f"[{A.name}] mood={A.mood.value} intensity={A.intensity:.2f} "
                  f"style={A.sinister_style:.1f} score={scores['score_a']:.1f}")
            print(f"[{B.name}] mood={B.mood.value} intensity={B.intensity:.2f} "
                  f"style={B.sinister_style:.1f} score={scores['score_b']:.1f}")


if __name__ == "__main__":
    random.seed(11)
    run_show(num_agents=12, rounds=4)

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

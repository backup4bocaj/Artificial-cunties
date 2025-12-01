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
    FROWN = "frown"
    SMILE = "smile"

@dataclass
class FaceParams:
    mouth_curve: float   # -5.0 (super frown) or +5.0 (super grin)
    brow_tilt: float     # -5.0 only (extreme downturned brows)
    eye_openness: float  # 5.0 only (cartoonishly wide eyes)
    thought_bubble: str  # stylized "thoughts" for cartoon effect

def exaggerated_face(mouth: MouthStyle) -> FaceParams:
    if mouth == MouthStyle.FROWN:
        return FaceParams(
            mouth_curve=-5.0,
            brow_tilt=-5.0,
            eye_openness=5.0,
            thought_bubble="Dark storm clouds… grumpy muttering… exaggerated negativity!"
        )
    else:
        return FaceParams(
            mouth_curve=+5.0,
            brow_tilt=-5.0,
            eye_openness=5.0,
            thought_bubble="Sunshine rays… hearts and sparkles… exaggerated positivity!"
        )

# Demo
if __name__ == "__main__":
    frown_face = exaggerated_face(MouthStyle.FROWN)
    smile_face = exaggerated_face(MouthStyle.SMILE)
    print("Extreme Frown:", frown_face)
    print("Extreme Smile:", smile_face)

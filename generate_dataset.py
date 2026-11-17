"""
generate_dataset.py  –  Photorealistic Leaf Dataset Generator
=============================================================
Creates 1000 images per class (128×128) that look like *real* crop
leaf photographs with natural colours, proper leaf shapes with pointed
tips, realistic vein networks, colour gradients, and authentic pest
damage patterns.

Usage:  python generate_dataset.py
"""
import os, math, numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
IMG_SIZE = (128, 128)
IMAGES_PER_CLASS = 1000
W, H = IMG_SIZE

# ═══════════════════════════════════════════════════════════════════════════
#  Helper – Natural Leaf Shape  (pointed ellipse with serrated edges)
# ═══════════════════════════════════════════════════════════════════════════

def _leaf_polygon(cx, cy, half_w, half_h, rng, serration=0, steps=120):
    """Return a list of (x, y) tuples forming a realistic leaf outline.

    The base shape is a *pointed* ellipse (narrow at tip & base) with
    optional serration (small teeth along the edge) for realism.
    """
    pts = []
    for i in range(steps):
        t = 2 * math.pi * i / steps
        # Pointed-ellipse envelope – narrower at 0° (tip) and 180° (base)
        sharpness = 0.55 + 0.45 * abs(math.sin(t))
        rx = half_w * sharpness
        ry = half_h

        x = cx + rx * math.cos(t)
        y = cy + ry * math.sin(t)

        # Serrated edge wobble
        if serration > 0:
            freq = rng.integers(10, 18)
            x += serration * math.sin(freq * t) * rng.uniform(0.6, 1.0)
            y += serration * math.cos(freq * t) * rng.uniform(0.4, 0.8)

        # Tiny organic wobble
        x += rng.uniform(-1.0, 1.0)
        y += rng.uniform(-1.0, 1.0)
        pts.append((x, y))
    return pts


# ═══════════════════════════════════════════════════════════════════════════
#  Helper – Realistic Vein Network
# ═══════════════════════════════════════════════════════════════════════════

def _draw_veins(draw, cx, cy, half_h, vein_color, rng):
    """Draw a central midrib + branching secondary/tertiary veins."""
    y_top = int(cy - half_h * 0.88)
    y_bot = int(cy + half_h * 0.88)

    # Central midrib (slight wobble)
    mid_pts = []
    for y in range(y_top, y_bot, 2):
        mid_pts.append((cx + rng.uniform(-0.8, 0.8), y))
    if len(mid_pts) >= 2:
        draw.line(mid_pts, fill=vein_color, width=2)

    # Secondary veins – alternate left & right
    num_sec = rng.integers(6, 12)
    for i in range(num_sec):
        frac = (i + 1) / (num_sec + 1)
        y_start = int(cy - half_h * 0.85 + frac * half_h * 1.7)
        direction = 1 if i % 2 == 0 else -1
        length = rng.integers(18, int(half_h * 0.7))
        angle_deg = rng.uniform(30, 60)
        angle = math.radians(angle_deg)

        # Curved vein (3-segment polyline)
        pts = [(cx, y_start)]
        x, y = cx, y_start
        seg_len = length / 3
        for s in range(3):
            curve = rng.uniform(-5, 5)
            x += direction * seg_len * math.cos(angle) + curve
            y -= seg_len * math.sin(angle) * rng.uniform(0.6, 1.0)
            pts.append((x, y))
        draw.line(pts, fill=vein_color, width=1)

        # Tertiary branches off secondary
        if rng.random() > 0.3:
            bx, by = pts[1]
            tx = bx + direction * rng.integers(6, 16)
            ty = by - rng.integers(4, 12)
            draw.line([(bx, by), (tx, ty)], fill=vein_color, width=1)
        if rng.random() > 0.5:
            bx, by = pts[2]
            tx = bx + direction * rng.integers(4, 12)
            ty = by - rng.integers(3, 9)
            draw.line([(bx, by), (tx, ty)], fill=vein_color, width=1)


# ═══════════════════════════════════════════════════════════════════════════
#  Helper – Natural Background  (soil / blurred green)
# ═══════════════════════════════════════════════════════════════════════════

def _natural_bg(rng):
    """Generate a soft, blurred natural-looking background."""
    bg_type = rng.choice(["soil", "field", "lab"])
    arr = np.zeros((H, W, 3), dtype=np.uint8)

    if bg_type == "soil":
        base = np.array([rng.integers(90, 140), rng.integers(70, 110),
                         rng.integers(40, 70)])
    elif bg_type == "field":
        base = np.array([rng.integers(60, 110), rng.integers(110, 160),
                         rng.integers(50, 90)])
    else:  # lab (neutral)
        v = rng.integers(180, 230)
        base = np.array([v, v, v])

    for c in range(3):
        arr[:, :, c] = base[c]
    # Add noise
    noise = rng.integers(-18, 18, (H, W, 3))
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(2.5, 5.0)))
    return img


# ═══════════════════════════════════════════════════════════════════════════
#  Helper – Draw base leaf on background
# ═══════════════════════════════════════════════════════════════════════════

def _base_leaf(rng, green_range, serration=1.5, vein_lightness=25):
    """Create a realistic leaf image on a natural background."""
    img = _natural_bg(rng)
    draw = ImageDraw.Draw(img)

    cx, cy = W // 2 + rng.integers(-6, 7), H // 2 + rng.integers(-6, 7)
    half_w = rng.integers(int(W * 0.28), int(W * 0.42))
    half_h = rng.integers(int(H * 0.36), int(H * 0.46))

    # Leaf colour – pick from natural green range
    g_lo, g_hi = green_range
    g = rng.integers(g_lo, g_hi)
    r = rng.integers(max(20, g - 90), max(40, g - 30))
    b = rng.integers(max(10, g - 100), max(30, g - 50))
    base_color = (r, g, b)
    border_color = tuple(np.clip(np.array(base_color) - 25, 0, 255))
    vein_color = tuple(np.clip(np.array(base_color) + vein_lightness, 0, 255))

    # Draw leaf shape
    leaf_pts = _leaf_polygon(cx, cy, half_w, half_h, rng, serration=serration)
    draw.polygon(leaf_pts, fill=base_color, outline=border_color)

    # Internal colour variation (subtle patches for realism)
    for _ in range(rng.integers(3, 7)):
        px = cx + rng.integers(-half_w // 2, half_w // 2)
        py = cy + rng.integers(-half_h // 2, half_h // 2)
        pr = rng.integers(6, 18)
        patch_c = tuple(np.clip(np.array(base_color) + rng.integers(-15, 15, 3), 0, 255))
        draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=patch_c)

    # Draw veins
    _draw_veins(draw, cx, cy, half_h, vein_color, rng)

    # Subtle light highlight
    hx = cx + rng.integers(-10, 10)
    hy = cy + rng.integers(-half_h // 3, 0)
    hr = rng.integers(8, 16)
    highlight = tuple(np.clip(np.array(base_color) + 30, 0, 255))
    draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=highlight)

    return img, draw, cx, cy, half_w, half_h, base_color


# ═══════════════════════════════════════════════════════════════════════════
#  Damage Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _add_yellowing(draw, cx, cy, hw, hh, rng, count=3):
    """Add yellow/brown discoloration patches."""
    for _ in range(count):
        px = cx + rng.integers(-hw // 2, hw // 2)
        py = cy + rng.integers(-hh // 2, hh // 2)
        pr = rng.integers(6, 16)
        yc = (rng.integers(160, 210), rng.integers(160, 200), rng.integers(30, 70))
        draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=yc)


def _add_tiny_dots(draw, cx, cy, hw, hh, rng, count_range=(25, 60)):
    """Add clusters of tiny dark dots (aphid bodies)."""
    n = rng.integers(*count_range)
    for _ in range(n):
        x = cx + rng.integers(-hw // 2, hw // 2)
        y = cy + rng.integers(-hh // 2, hh // 2)
        r = rng.integers(1, 3)
        c = (rng.integers(20, 60), rng.integers(30, 60), rng.integers(5, 25))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=c)


def _add_chew_holes(img, draw, cx, cy, hw, hh, rng, count_range=(3, 8)):
    """Cut out irregular holes (caterpillar / beetle damage).
    Fills with background colour sampled from corners."""
    arr = np.array(img)
    # Sample background colour from corner
    bg_sample = arr[3:8, 3:8].mean(axis=(0, 1)).astype(np.uint8)
    bg_c = tuple(bg_sample)
    n = rng.integers(*count_range)
    for _ in range(n):
        x = cx + rng.integers(-hw * 2 // 3, hw * 2 // 3)
        y = cy + rng.integers(-hh * 2 // 3, hh * 2 // 3)
        rw = rng.integers(4, 14)
        rh = rng.integers(3, 11)
        draw.ellipse([x - rw, y - rh, x + rw, y + rh], fill=bg_c)
        # Dark edge ring around hole
        edge_c = tuple(np.clip(np.array(bg_c) - 30, 0, 255))
        draw.ellipse([x - rw - 1, y - rh - 1, x + rw + 1, y + rh + 1], outline=edge_c)


def _add_brown_streaks(draw, cx, cy, hw, hh, rng, count=6):
    """Add elongated brown streaks (caterpillar feeding trails)."""
    for _ in range(count):
        x1 = cx + rng.integers(-hw // 2, hw // 2)
        y1 = cy + rng.integers(-hh // 2, hh // 2)
        length = rng.integers(12, 35)
        angle = rng.uniform(0, math.pi)
        x2 = x1 + int(length * math.cos(angle))
        y2 = y1 + int(length * math.sin(angle))
        c = (rng.integers(100, 150), rng.integers(60, 100), rng.integers(20, 50))
        draw.line([(x1, y1), (x2, y2)], fill=c, width=rng.integers(2, 4))


def _add_dark_spots(draw, cx, cy, hw, hh, rng, count_range=(6, 16)):
    """Add medium dark spots (beetle frass / damage markers)."""
    n = rng.integers(*count_range)
    for _ in range(n):
        x = cx + rng.integers(-hw // 2, hw // 2)
        y = cy + rng.integers(-hh // 2, hh // 2)
        r = rng.integers(3, 8)
        c = (rng.integers(30, 70), rng.integers(25, 55), rng.integers(10, 35))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=c)


def _add_edge_bites(img, draw, cx, cy, hw, hh, rng, count=5):
    """Cut irregular chunks from leaf edges (grasshopper feeding)."""
    arr = np.array(img)
    bg_sample = arr[3:8, 3:8].mean(axis=(0, 1)).astype(np.uint8)
    bg_c = tuple(bg_sample)
    for _ in range(count):
        side = rng.choice(4)
        if side == 0:    # left
            x, y = cx - hw + rng.integers(-5, 10), cy + rng.integers(-hh // 2, hh // 2)
        elif side == 1:  # right
            x, y = cx + hw + rng.integers(-10, 5), cy + rng.integers(-hh // 2, hh // 2)
        elif side == 2:  # top
            x, y = cx + rng.integers(-hw // 2, hw // 2), cy - hh + rng.integers(-5, 10)
        else:            # bottom
            x, y = cx + rng.integers(-hw // 2, hw // 2), cy + hh + rng.integers(-10, 5)
        rw = rng.integers(8, 22)
        rh = rng.integers(6, 18)
        draw.ellipse([x - rw, y - rh, x + rw, y + rh], fill=bg_c)


# ═══════════════════════════════════════════════════════════════════════════
#  Post-processing – make it look photographed
# ═══════════════════════════════════════════════════════════════════════════

def _photo_finish(img, rng):
    """Apply slight blur, brightness/contrast jitter, and noise to
    simulate a real phone/camera photograph."""
    # Soft focus
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))

    # Brightness & contrast jitter
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(rng.uniform(0.88, 1.12))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(rng.uniform(0.90, 1.10))

    # Sensor noise
    arr = np.array(img, dtype=np.int16)
    noise = rng.integers(-8, 8, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ═══════════════════════════════════════════════════════════════════════════
#  Per-class Generators
# ═══════════════════════════════════════════════════════════════════════════

def make_healthy(rng):
    img, draw, cx, cy, hw, hh, bc = _base_leaf(
        rng, green_range=(130, 200), serration=rng.uniform(0, 1.5))
    return _photo_finish(img, rng)


def make_aphids(rng):
    # Slightly yellowed leaf
    img, draw, cx, cy, hw, hh, bc = _base_leaf(
        rng, green_range=(120, 175), serration=rng.uniform(0.5, 2.0))
    _add_yellowing(draw, cx, cy, hw, hh, rng, count=rng.integers(2, 5))
    _add_tiny_dots(draw, cx, cy, hw, hh, rng, count_range=(30, 70))
    return _photo_finish(img, rng)


def make_armyworm(rng):
    img, draw, cx, cy, hw, hh, bc = _base_leaf(
        rng, green_range=(110, 170), serration=rng.uniform(0.5, 2.0))
    _add_chew_holes(img, draw, cx, cy, hw, hh, rng, count_range=(4, 9))
    _add_brown_streaks(draw, cx, cy, hw, hh, rng, count=rng.integers(5, 12))
    return _photo_finish(img, rng)


def make_beetles(rng):
    img, draw, cx, cy, hw, hh, bc = _base_leaf(
        rng, green_range=(105, 165), serration=rng.uniform(1.0, 2.5))
    _add_chew_holes(img, draw, cx, cy, hw, hh, rng, count_range=(5, 12))
    _add_dark_spots(draw, cx, cy, hw, hh, rng, count_range=(8, 20))
    return _photo_finish(img, rng)


def make_grasshopper(rng):
    img, draw, cx, cy, hw, hh, bc = _base_leaf(
        rng, green_range=(115, 175), serration=rng.uniform(0.5, 2.0))
    _add_edge_bites(img, draw, cx, cy, hw, hh, rng, count=rng.integers(4, 8))
    # Brown discoloration at bite edges
    _add_yellowing(draw, cx, cy, hw, hh, rng, count=rng.integers(2, 4))
    return _photo_finish(img, rng)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

GENERATORS = {
    "Healthy":     make_healthy,
    "Aphids":      make_aphids,
    "Armyworm":    make_armyworm,
    "Beetles":     make_beetles,
    "Grasshopper": make_grasshopper,
}


def main():
    total = 0
    for name, gen_fn in GENERATORS.items():
        d = os.path.join(DATASET_DIR, name)
        os.makedirs(d, exist_ok=True)
        # Remove old images
        for f in os.listdir(d):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                os.remove(os.path.join(d, f))
        print(f"[GEN] {name}: generating {IMAGES_PER_CLASS} photo-realistic leaf images ...")
        for i in range(IMAGES_PER_CLASS):
            rng = np.random.default_rng(seed=abs(hash(name)) + i)
            img = gen_fn(rng)
            img.save(os.path.join(d, f"IMG_{i + 1:04d}.png"))
        total += IMAGES_PER_CLASS
        print(f"      [OK] {name} done.")
    print(f"\n[DONE] Dataset generation complete! ({total} total images)")
    print("   Images now resemble real photographed crop leaves.")


if __name__ == "__main__":
    main()

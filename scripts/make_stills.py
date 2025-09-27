# make_stills.py
# Hard-coded inputs and selections so you can just:  python make_stills.py
# Requires: pip install pillow imageio

from PIL import Image, ImageDraw
import imageio.v2 as imageio
import os

# ---- Hard-coded settings ----
GIF_MAIN = "imageData.gif"          # your exported 3x6 grid GIF
GIF_LATE = "imageData_late.gif"     # optional, for early vs late comparison
ROWS, COLS = 3, 6                    # grid layout: 3 rows x 6 cols
ROW_TRUTH, ROW_ERROR, ROW_MODEL = 0, 1, 2
COL = 4                              # which example column to use (0..5)
FRAMES = [4, 5, 6, 10, 20, 40, 60]   # a small, readable subset (63 total in your GIF)
MARK_AFTER = 5                       # draw divider after this frame index in the timeline
PAD_H, PAD_V = 2, 4                  # pixel padding between tiles

# ---- Helpers ----
def load_gif(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    frames = imageio.mimread(path)
    if not frames:
        raise ValueError(f"No frames in {path}")
    return [Image.fromarray(f) for f in frames]

def crop_cell(img, row, col):
    W, H = img.size
    cw, ch = W // COLS, H // ROWS
    x0, y0 = col * cw, row * ch
    return img.crop((x0, y0, x0 + cw, y0 + ch))

def concat_h(imgs, pad=0):
    h = imgs[0].height
    w = sum(im.width for im in imgs) + pad * (len(imgs) - 1)
    out = Image.new("RGB", (w, h))
    x = 0
    for i, im in enumerate(imgs):
        out.paste(im, (x, 0))
        x += im.width + pad
    return out

def stack_v(imgs, pad=0):
    w = max(im.width for im in imgs)
    h = sum(im.height for im in imgs) + pad * (len(imgs) - 1)
    out = Image.new("RGB", (w, h))
    y = 0
    for im in imgs:
        out.paste(im, (0, y))
        y += im.height + pad
    return out

def safe_frames(frames, n):
    # keep only indices that exist
    return [i for i in frames if 0 <= i < n]

def draw_divider(strip, cells, boundary_index):
    if boundary_index < 0 or boundary_index >= len(cells) - 1:
        return
    x = sum(c.width for c in cells[:boundary_index + 1])
    draw = ImageDraw.Draw(strip)
    draw.line([(x + 1, 0), (x + 1, strip.height)], fill=(255, 0, 0), width=1)

# ---- Main outputs ----
def make_timeline(gif_path, out_png):
    frames = load_gif(gif_path)
    n = len(frames)
    chosen = safe_frames(FRAMES, n)
    truth_cells = [crop_cell(frames[i], ROW_TRUTH, COL) for i in chosen]
    model_cells = [crop_cell(frames[i], ROW_MODEL, COL) for i in chosen]
    top = concat_h(truth_cells, pad=PAD_H)
    bottom = concat_h(model_cells, pad=PAD_H)
    # divider after MARK_AFTER if that frame is included
    if MARK_AFTER in chosen:
        idx = chosen.index(MARK_AFTER)
        draw_divider(top, truth_cells, idx)
        draw_divider(bottom, model_cells, idx)
    out = stack_v([top, bottom], pad=PAD_V)
    out.save(out_png)
    print("wrote", out_png)

def make_error_frame(gif_path, frame_idx, out_png):
    frames = load_gif(gif_path)
    if not (0 <= frame_idx < len(frames)):
        raise ValueError(f"frame {frame_idx} out of range 0..{len(frames)-1}")
    cell = crop_cell(frames[frame_idx], ROW_ERROR, COL)
    cell.save(out_png)
    print("wrote", out_png)

def maybe_compare_error(early_gif, late_gif, frame_idx, out_early, out_late, out_side_by_side):
    if not os.path.exists(late_gif):
        return
    make_error_frame(early_gif, frame_idx, out_early)
    make_error_frame(late_gif, frame_idx, out_late)
    img1 = Image.open(out_early)
    img2 = Image.open(out_late)
    side = concat_h([img1, img2], pad=8)
    side.save(out_side_by_side)
    print("wrote", out_side_by_side)

if __name__ == "__main__":
    # 1) Timeline from main GIF
    make_timeline(GIF_MAIN, "timeline.png")
    # 2) Error strip (single frame) from main GIF
    make_error_frame(GIF_MAIN, 6, "error_frame6.png")
    # 3) Optional early vs late comparison if a second GIF exists
    maybe_compare_error(GIF_MAIN, GIF_LATE, 6,
                        "error_frame6_early.png",
                        "error_frame6_late.png",
                        "error_compare.png")

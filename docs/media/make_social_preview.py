"""Render docs/media/social-preview.png (1280x640) for the GitHub repository social preview.

Run from the repo root:

    uv run --no-project --with pillow python docs/media/make_social_preview.py

Uses system fonts only (Helvetica Neue, falling back to Arial). No third-party logos.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640
OUT = Path(__file__).with_name("social-preview.png")

BG = (14, 16, 21)
INK = (242, 243, 245)
MUTED = (160, 166, 178)
ACCENT = (88, 166, 255)
RULE = (44, 48, 58)

TAGLINE = "Talk to your Mac. It does the work, in your apps, and asks before anything leaves the machine."
FEATURES = ("Voice first", "Works in your apps", "Asks before it acts")
FOOTER = "MIT  ·  macOS 14+  ·  Apple Silicon  ·  local speech"


def font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    ttc = Path("/System/Library/Fonts/HelveticaNeue.ttc")
    if ttc.exists():
        want = "Bold" if bold else "Regular"
        for index in range(0, 24):
            try:
                f = ImageFont.truetype(str(ttc), size, index=index)
            except OSError:
                break
            family, style = f.getname()
            if family == "Helvetica Neue" and style == want:
                return f
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=f) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def main() -> int:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # A soft accent glow top-right, drawn as concentric translucent circles.
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g = ImageDraw.Draw(glow)
    cx, cy = 1120, 120
    for r in range(420, 0, -12):
        a = int(26 * (1 - r / 420) ** 1.6)
        g.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*ACCENT, a))
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
    draw = ImageDraw.Draw(img)

    margin = 88
    # Small label line.
    draw.text((margin, 86), "OPEN SOURCE  ·  VOICE AGENT FOR macOS", font=font(22, True), fill=ACCENT)

    # Title.
    title_f = font(148, True)
    draw.text((margin - 6, 112), "OpenValet", font=title_f, fill=INK)

    # Tagline, wrapped to the text column.
    tag_f = font(34, False)
    y = 300
    for line in wrap(draw, TAGLINE, tag_f, W - 2 * margin - 80):
        draw.text((margin, y), line, font=tag_f, fill=MUTED)
        y += 46

    # Feature words with a dot between them.
    feat_f = font(30, True)
    y = 436
    x = margin
    for i, word in enumerate(FEATURES):
        if i:
            draw.ellipse((x + 14, y + 16, x + 22, y + 24), fill=ACCENT)
            x += 50
        draw.text((x, y), word, font=feat_f, fill=INK)
        x += draw.textlength(word, font=feat_f)

    # Rule and footer.
    draw.line((margin, 540, W - margin, 540), fill=RULE, width=2)
    draw.text((margin, 562), FOOTER, font=font(24, False), fill=MUTED)
    repo_f = font(24, False)
    repo = "github.com/SamarthY06/OpenValet"
    draw.text((W - margin - draw.textlength(repo, font=repo_f), 562), repo, font=repo_f, fill=MUTED)

    img.save(OUT, optimize=True)
    print(OUT, img.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())

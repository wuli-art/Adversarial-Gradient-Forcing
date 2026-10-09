"""Build the homepage comparison from the six distributed case-049 videos.

Requires Pillow and ffmpeg (or imageio-ffmpeg). Frames retain their full view.
The timeline follows the paper's nominal 0s, 48s, ..., 240s convention; the
240s endpoint uses the last available frame at 239.75s in the web video.
Run from any directory: python project-page/figures/build_yawning_comparison.py
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
METHODS = [('agf', 'AGF (ours)'), ('self-forcing', 'Self-Forcing'),
           ('causal-forcing', 'Causal Forcing'), ('longlive', 'LongLive'),
           ('sgf', 'SGF*'), ('opsdv', 'OPSD-V†')]
TIMES = [0, 48, 96, 144, 192, 239.75]
LABELS = ['0s', '48s', '96s', '144s', '192s', '240s']
FFMPEG = shutil.which('ffmpeg')
if not FFMPEG:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def font(size, weight=400):
    face = ImageFont.truetype(str(ROOT / 'assets/fonts/inter-variable.ttf'), size)
    face.set_variation_by_axes([24, weight])
    return face


def main():
    margin, label_width, gap, cell_w, cell_h = 24, 408, 8, 624, 360
    top, row_gap, bottom = 126, 12, 80
    width = margin * 2 + label_width + 6 * cell_w + 5 * gap
    height = top + 6 * cell_h + 5 * row_gap + bottom
    ink, muted, accent = '#23262b', '#59616b', '#304bb7'
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    regular, bold, small = font(49), font(48, 600), font(42)
    x0 = margin + label_width
    for col, label in enumerate(LABELS):
        x = x0 + col * (cell_w + gap) + cell_w / 2
        draw.text((x, 31), label, font=regular, fill=muted, anchor='mt')
    # A single left-to-right time arrow, matching Figures 1 and 5 of the paper.
    arrow_y, arrow_end = top - 23, width - margin
    draw.line((x0, arrow_y, arrow_end - 27, arrow_y), fill=ink, width=6)
    draw.polygon([(arrow_end, arrow_y), (arrow_end - 32, arrow_y - 16),
                  (arrow_end - 32, arrow_y + 16)], fill=ink)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        def extract(job):
            method, col, timestamp = job
            output = tmp / f'{method}-{col}.png'
            subprocess.run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-y',
                            '-ss', str(timestamp), '-i', str(ROOT / f'assets/videos/049/{method}.mp4'),
                            '-frames:v', '1', str(output)], check=True)
            return method, col, output
        jobs = [(method, col, t) for method, _ in METHODS for col, t in enumerate(TIMES)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            for method, col, output in pool.map(extract, jobs):
                row = [m for m, _ in METHODS].index(method)
                with Image.open(output) as frame:
                    assert frame.size == (cell_w, cell_h)
                    canvas.paste(frame, (x0 + col * (cell_w + gap), top + row * (cell_h + row_gap)))
        for row, (method, label) in enumerate(METHODS):
            y = top + row * (cell_h + row_gap)
            draw.rounded_rectangle((margin, y, x0 - 12, y + cell_h), radius=10,
                                   fill='#f0f3fc' if method == 'agf' else '#fafbfc',
                                   outline='#d9dfe8', width=2)
            draw.text((margin + (label_width - 12) / 2, y + cell_h / 2), label,
                      font=bold, fill=accent if method == 'agf' else ink, anchor='mm')
    data = (ROOT / 'data.js').read_text().split('window.AGF_CASES = ', 1)[1].strip().removesuffix(';')
    prompt = next(case['prompt'] for case in json.loads(data) if case['id'] == '049')
    caption = 'Prompt: ' + prompt.split('. ')[0] + '.'
    draw.text((x0, height - 35), caption, font=small, fill=ink, anchor='lm')
    out = ROOT / 'assets/yawning-child-comparison.png'
    canvas.save(out, optimize=True)
    preview = canvas.resize((2544, round(height * 2544 / width)), Image.Resampling.LANCZOS)
    preview.save(out.with_suffix('.webp'), lossless=True, method=6)
    print(f'Saved PNG {width} × {height}; WebP {preview.width} × {preview.height}')


if __name__ == '__main__':
    main()

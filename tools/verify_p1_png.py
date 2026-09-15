"""Verify reconstructed PNGs and compare selected frames with the FP16 source."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

root = Path(sys.argv[1])
pngs = sorted((root / 'reconstruction').glob('im*.png'))
if [p.name for p in pngs] != ['im{:05d}.png'.format(i + 1) for i in range(120)]:
    raise RuntimeError('expected exactly 120 consecutive PNGs')
for path in pngs:
    with Image.open(path) as im:
        im.load()
        if im.size != (1920, 1080) or im.mode != 'RGB':
            raise RuntimeError('invalid reconstruction: ' + str(path))

results = []
preview = Image.new('RGB', (1280, 4 * 390), 'white')
draw = ImageDraw.Draw(preview)
for row, index in enumerate([0, 32, 96, 119]):
    values = np.fromfile(root / 'source' / ('frame_{}.fp16'.format(index)), dtype=np.float16)
    y, cb, cr = values.astype(np.float32).reshape(3, 1088, 1920)[:, :1080, :]
    red = y + 1.5748 * (cr - 0.5)
    blue = y + 1.8556 * (cb - 0.5)
    green = (y - 0.2126 * red - 0.0722 * blue) / 0.7152
    source = np.rint(np.clip(np.stack((red, green, blue), axis=2), 0, 1) * 255).astype(np.uint8)
    with Image.open(root / 'reconstruction' / ('im{:05d}.png'.format(index + 1))) as im:
        decoded = np.array(im)
    mse = np.mean((source.astype(np.float64) - decoded.astype(np.float64)) ** 2)
    psnr = float(10 * np.log10(255.0 ** 2 / mse)) if mse else None
    results.append({'frame_index': index, 'rgb_psnr_db': psnr,
                    'reconstruction_std': float(decoded.std())})
    draw.text((8, row * 390 + 8), 'Frame {}: source (left) / decoded (right), PSNR {:.2f} dB'.format(index, psnr), fill='black')
    preview.paste(Image.fromarray(source).resize((640, 360)), (0, row * 390 + 30))
    preview.paste(Image.fromarray(decoded).resize((640, 360)), (640, row * 390 + 30))
preview.save(root / 'source-vs-decoded.jpg', quality=92)
report = {'png_count': len(pngs), 'all_readable': True, 'size': [1920, 1080],
          'sample_rgb_psnr': results, 'note': 'Four sampled frames; not a full-sequence quality benchmark.'}
(root / 'image-verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

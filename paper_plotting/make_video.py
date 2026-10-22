"""Side-by-side comparison video for the RA-L submission (DDIM-1 | STEP-1 | Ours-1, same initial state).

    python make_video.py --manifest data/v2/raw/eval/sim_pusht/manifest.json \
        --video_root data/v2/raw/eval/sim_pusht --seeds 100014 100003 100002 100006 --out ../video/pusht_one_step.mp4
"""
import argparse, json, os
import numpy as np
import cv2
import imageio.v2 as imageio

ROWS = [('ddim', 'DDIM, 1 NFE'), ('step', 'STEP, 1 NFE'), ('ours', 'Ours, 1 NFE')]
S = 3          # upscale factor (96 -> 288 px); --scale overrides
HEAD, FOOT, PAD = 46, 40, 12
GOLD = (0, 192, 255)  # BGR for cv2 drawing on RGB frames is handled below (we draw in RGB)


def frames(path):
    r = imageio.get_reader(path); f = [x for x in r]; r.close(); return f


def put(img, text, org, scale=0.7, color=(0, 0, 0), bold=False):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if bold else 1, cv2.LINE_AA)


def card(w, h, lines, n=20):
    img = np.full((h, w, 3), 255, np.uint8)
    for i, (t, sc, bold) in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, 2 if bold else 1)
        put(img, t, ((w - tw) // 2, h // 2 - 30 + 40 * i), sc, bold=bold)
    return [img] * n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--video_root', required=True)
    p.add_argument('--seeds', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--scale', type=int, default=None)
    p.add_argument('--title', default='Push-T, one network evaluation per action chunk')
    p.add_argument('--metric', default='coverage')
    a = p.parse_args()
    man = json.load(open(a.manifest))
    global S
    vids0 = man['ours'][a.seeds[0]]['video']
    vkey = '/' + os.path.basename(os.path.normpath(a.video_root)) + '/'
    resolve = lambda v: v if os.path.exists(v) else os.path.join(a.video_root, v.split(vkey)[-1])
    h0 = imageio.get_reader(resolve(vids0)).get_data(0).shape[0]
    S = a.scale or max(1, round(288 / h0))
    tile = h0 * S
    W = PAD + len(ROWS) * (tile + PAD)
    H = HEAD + tile + FOOT
    out = [] + card(W, H, [(a.title, 0.75, True),
                           ('same initial state for all three methods', 0.6, False),
                           ('frozen Diffusion Policy; only the start of denoising differs', 0.6, False)], 50)
    for k, seed in enumerate(a.seeds):
        vids = []
        for key, _ in ROWS:
            vids.append(frames(resolve(man[key][seed]['video'])))
        L = max(len(f) for f in vids)
        for t in range(L):
            img = np.full((H, W, 3), 255, np.uint8)
            for i, ((key, lab), fr) in enumerate(zip(ROWS, vids)):
                x0 = PAD + i * (tile + PAD)
                f = cv2.resize(fr[min(t, len(fr) - 1)], (tile, tile), interpolation=cv2.INTER_NEAREST)
                img[HEAD:HEAD + tile, x0:x0 + tile] = f
                col = (255, 192, 0) if key == 'ours' else (127, 127, 127)
                cv2.rectangle(img, (x0 - 2, HEAD - 2), (x0 + tile + 1, HEAD + tile + 1), col, 3 if key == 'ours' else 1)
                put(img, lab, (x0 + 6, HEAD - 14), 0.62, bold=(key == 'ours'))
                if t == L - 1 or t > L - 25:
                    r = man[key][seed]['reward']
                    put(img, (f'{a.metric} {r:.2f}' if a.metric == 'coverage' else ('success' if r > 0.5 else 'failure')), (x0 + 6, HEAD + tile + 28), 0.6,
                        (192, 0, 0) if key == 'ours' else (0, 0, 0), bold=(key == 'ours'))
            put(img, f'episode {k + 1}/{len(a.seeds)}   step {t:3d}', (W - 260, H - 10), 0.45, (90, 90, 90))
            out.append(img)
        out += [out[-1]] * 20
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    imageio.mimwrite(a.out, out, fps=a.fps, codec='libx264', quality=8, macro_block_size=1)
    print(a.out, len(out), 'frames', f'{len(out) / a.fps:.1f}s')


if __name__ == '__main__':
    main()

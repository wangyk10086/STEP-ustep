"""Parallel re-rendering of robomimic image observations from simulator states.
usage: python prender.py SRC.hdf5 DST.hdf5 NPROC"""
import os, sys, shutil, multiprocessing as mp
os.environ['MUJOCO_GL'] = 'egl'
import numpy as np, h5py

SRC, DST, NP = sys.argv[1], sys.argv[2], int(sys.argv[3])
CAMS = ['agentview', 'robot0_eye_in_hand']
TMP = DST + '.parts'


def work(args):
    wid, demos = args
    import robomimic.utils.file_utils as FU, robomimic.utils.env_utils as EU, robomimic.utils.obs_utils as OU
    OU.initialize_obs_modality_mapping_from_dict({'low_dim': ['robot0_eef_pos'], 'rgb': [c + '_image' for c in CAMS]})
    meta = FU.get_env_metadata_from_dataset(SRC)
    env = EU.create_env_for_data_processing(env_meta=meta, camera_names=CAMS, camera_height=84, camera_width=84,
                                            reward_shaping=False)
    f = h5py.File(SRC, 'r')
    for d in demos:
        out = os.path.join(TMP, d + '.npz')
        if os.path.exists(out):
            continue
        states = f[f'data/{d}/states'][()]
        init = dict(states=states[0])
        if 'model_file' in f[f'data/{d}'].attrs:
            init['model'] = f[f'data/{d}'].attrs['model_file']
        env.reset()
        env.reset_to(init)
        imgs = {c: [] for c in CAMS}
        for t in range(states.shape[0]):
            o = env.reset_to({'states': states[t]})
            for c in CAMS:
                imgs[c].append(o[c + '_image'])
        np.savez(out + '.tmp.npz', **{c: np.stack(v).astype(np.uint8) for c, v in imgs.items()})
        os.replace(out + '.tmp.npz', out)
        print(wid, d, flush=True)
    return wid


if __name__ == '__main__':
    os.makedirs(TMP, exist_ok=True)
    with h5py.File(SRC, 'r') as f:
        demos = sorted(f['data'].keys(), key=lambda s: int(s.split('_')[1]))
    chunks = [(i, demos[i::NP]) for i in range(NP)]
    with mp.get_context('spawn').Pool(NP) as pool:
        for w in pool.imap_unordered(work, chunks):
            print('worker done', w, flush=True)
    shutil.copy(SRC, DST + '.tmp')
    with h5py.File(DST + '.tmp', 'a') as g:
        for d in demos:
            z = np.load(os.path.join(TMP, d + '.npz'))
            n = g[f'data/{d}/actions'].shape[0]
            for c in CAMS:
                x = z[c]
                assert x.shape[0] == n, (d, x.shape, n)
                # robomimic obs images are HWC uint8; flip handled by env (same as dataset_states_to_obs)
                g.create_dataset(f'data/{d}/obs/{c}_image', data=x, compression='gzip', compression_opts=1)
    os.replace(DST + '.tmp', DST)
    print('MERGE_DONE', DST, flush=True)

"""
Aligned previous-chunk wrapper (U-STEP).

The wrapper used by STEP's training workspaces
(action_predictor_lowdim_policy.ActionPredictorDatasetWrapper) sets

    prev_action = action[:prev_action_horizon]

i.e. the first steps of the *same* window that is also the regression target.
The predictor can then minimize MSE on those steps by copying its input.  At
inference, however, prev_action is the chunk produced at the *previous* call,
which covers times shifted by -n_action_steps.  The model therefore copies
stale (past) actions into the head of the new chunk -- a copycat failure that
is consistent with the execution stalls STEP patches with velocity-aware
perturbation.

This wrapper instead returns, for each window starting at virtual time v0, the
ground-truth chunk that the previous inference call would have produced:

    prev_action = actions[v0 - shift : v0 - shift + prev_action_horizon]

with shift = n_action_steps (executed steps per call), clamped/padded at
episode boundaries exactly like SequenceSampler, plus `prev_valid` = 0 when no
previous call exists yet (episode start), in which case the predictor sees
zeros in normalized space -- matching inference.
"""

from typing import Dict
import numpy as np
import torch

from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer


class AlignedPrevActionWrapper(BaseLowdimDataset):
    def __init__(
        self,
        base_dataset,
        prev_action_horizon: int,
        n_action_steps: int,
        n_obs_steps: int = 2,
        action_key: str = None,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.prev_action_horizon = prev_action_horizon
        self.shift = n_action_steps
        self.n_obs_steps = n_obs_steps
        if action_key is None:
            action_key = getattr(base_dataset, 'action_key', 'action')
        self.action_key = action_key

        sampler = getattr(base_dataset, 'sampler', None)
        rb = getattr(base_dataset, 'replay_buffer', None)
        if sampler is None or rb is None:
            raise ValueError('AlignedPrevActionWrapper needs base_dataset.sampler and .replay_buffer')
        self._indices = np.asarray(sampler.indices)
        self._actions = np.asarray(rb[self.action_key])
        self._episode_ends = np.asarray(rb.episode_ends)
        self._episode_starts = np.concatenate([[0], self._episode_ends[:-1]])

    def __len__(self):
        return len(self.base_dataset)

    def _prev_chunk(self, idx: int):
        buffer_start, _, sample_start, _ = self._indices[idx]
        v0 = int(buffer_start) - int(sample_start)
        ep = int(np.searchsorted(self._episode_ends, buffer_start, side='right'))
        ep_start = int(self._episode_starts[ep])
        ep_end = int(self._episode_ends[ep])
        p0 = v0 - self.shift
        # first inference call of an episode happens at v0 = ep_start - (To - 1)
        valid = p0 >= ep_start - (self.n_obs_steps - 1)
        t = np.arange(p0, p0 + self.prev_action_horizon)
        t = np.clip(t, ep_start, ep_end - 1)
        return self._actions[t], valid

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base_dataset[idx]
        prev, valid = self._prev_chunk(idx)
        ref = sample['action']
        sample['prev_action'] = torch.as_tensor(prev, dtype=ref.dtype)
        sample['prev_valid'] = torch.tensor(1.0 if valid else 0.0, dtype=ref.dtype)
        return sample

    def get_normalizer(self, **kwargs):
        normalizer = self.base_dataset.get_normalizer(**kwargs)
        normalizer['prev_action'] = normalizer['action']
        if 'obs' not in normalizer.params_dict:
            normalizer['obs'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def get_validation_dataset(self):
        return AlignedPrevActionWrapper(
            self.base_dataset.get_validation_dataset(),
            self.prev_action_horizon, self.shift,
            n_obs_steps=self.n_obs_steps, action_key=self.action_key)

    def get_all_actions(self) -> torch.Tensor:
        return self.base_dataset.get_all_actions()

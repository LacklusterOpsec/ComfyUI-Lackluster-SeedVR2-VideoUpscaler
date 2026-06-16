# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

"""
DPM-Solver++ 2M sampler for Flow Matching ODEs.
"""

from typing import Callable
import torch

from ..types import PredictionType
from ..utils import expand_dims
from .base import Sampler, SamplerModelArgs


class DPMSolverSampler(Sampler):
    """
    DPM-Solver++ 2M sampler adapted for Flow Matching ODEs.
    It corresponds to Adams-Bashforth 2nd order method.
    """

    def sample(
        self,
        x: torch.Tensor,
        f: Callable[[SamplerModelArgs], torch.Tensor],
    ) -> torch.Tensor:
        timesteps = self.timesteps.timesteps
        progress = self.get_progress_bar()
        i = 0
        
        last_v = None
        last_h = None

        for t, s in zip(timesteps[:-1], timesteps[1:]):
            pred = f(SamplerModelArgs(x, t, i))
            
            # For flow matching (v_lerp), pred is v = x_T - x_0.
            # We can also compute v directly from pred_x_0 and pred_x_T.
            pred_x_0, pred_x_T = self.schedule.convert_from_pred(pred, self.prediction_type, x, t)
            v = pred_x_T - pred_x_0
            
            h = s - t
            
            if last_v is None or i == 0:
                # First step: use Euler
                dx = h * v
            else:
                # DPM-Solver++ 2M / Adams-Bashforth 2
                r = h / last_h
                # v_eff = (1 + r/2) * v - (r/2) * last_v
                v_eff = v + (r / 2.0) * (v - last_v)
                dx = h * v_eff
                
            # Expand dx to match x
            dx = expand_dims(dx, x.ndim)
            
            x = x + dx
            
            last_v = v.clone()
            last_h = h.clone() if isinstance(h, torch.Tensor) else h
            
            del pred, pred_x_0, pred_x_T, v, h, dx
            
            i += 1
            progress.update()

        if self.return_endpoint:
            t = timesteps[-1]
            pred = f(SamplerModelArgs(x, t, i))
            x = self.get_endpoint(pred, x, t)
            del pred
            progress.update()
            
        return x

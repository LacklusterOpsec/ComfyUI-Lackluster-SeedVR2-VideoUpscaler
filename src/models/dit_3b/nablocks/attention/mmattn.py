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

from typing import Optional, Tuple, Union
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _triple

from .....common.cache import Cache
from .....common.distributed.ops import gather_heads_scatter_seq, gather_seq_scatter_heads_qkv
from .....common.half_precision_fixes import safe_pad_operation

from ... import na
from ...attention import FlashAttentionVarlen
from ...mm import MMArg, MMModule
from ...normalization import norm_layer_type
from ...rope import get_na_rope
from ...window import get_window_op
from itertools import chain


class NaMMAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        rope_type: Optional[str],
        rope_dim: int,
        shared_weights: bool,
        attention_mode: str = 'sdpa',
        **kwargs,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        inner_dim = heads * head_dim
        qkv_dim = inner_dim * 3
        self.head_dim = head_dim
        self.proj_qkv = MMModule(
            nn.Linear, dim, qkv_dim, bias=qk_bias, shared_weights=shared_weights
        )
        self.proj_out = MMModule(nn.Linear, inner_dim, dim, shared_weights=shared_weights)
        self.norm_q = MMModule(
            qk_norm,
            dim=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
        )
        self.norm_k = MMModule(
            qk_norm,
            dim=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
        )

        self.rope = get_na_rope(rope_type=rope_type, dim=rope_dim)
        self.attn = FlashAttentionVarlen(attention_mode=attention_mode)

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        vid_qkv = gather_seq_scatter_heads_qkv(
            vid_qkv,
            seq_dim=0,
            qkv_shape=vid_shape,
            cache=cache.namespace("vid"),
        )
        txt_qkv = gather_seq_scatter_heads_qkv(
            txt_qkv,
            seq_dim=0,
            qkv_shape=txt_shape,
            cache=cache.namespace("txt"),
        )
        vid_qkv = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        if self.rope:
            if self.rope.mm:
                vid_q, vid_k, txt_q, txt_k = self.rope(
                    vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache
                )
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, vid_shape, cache)

        vid_len = cache("vid_len", lambda: vid_shape.prod(-1))
        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))
        all_len = cache("all_len", lambda: vid_len + txt_len)

        concat, unconcat = cache("mm_pnp", lambda: na.concat_idx(vid_len, txt_len))

        # Attention handles dtype conversion internally using pipeline compute_dtype
        attn = self.attn(
            q=concat(vid_q, txt_q),
            k=concat(vid_k, txt_k),
            v=concat(vid_v, txt_v),
            cu_seqlens_q=cache("mm_seqlens", lambda: safe_pad_operation(all_len.cumsum(0), (1, 0)).int()),
            cu_seqlens_k=cache("mm_seqlens", lambda: safe_pad_operation(all_len.cumsum(0), (1, 0)).int()),
            max_seqlen_q=cache("mm_maxlen", lambda: all_len.max()),
            max_seqlen_k=cache("mm_maxlen", lambda: all_len.max()),
        ).type_as(vid_q)

        attn = rearrange(attn, "l h d -> l (h d)")
        vid_out, txt_out = unconcat(attn)
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out


class NaSwinAttention(NaMMAttention):
    def __init__(
        self,
        *args,
        window: Union[int, Tuple[int, int, int]],
        window_method: str,
        attention_mode: str = 'sdpa',
        **kwargs,
    ):
        super().__init__(*args, attention_mode=attention_mode, **kwargs)
        self.window = _triple(window)
        self.window_method = window_method
        assert all(map(lambda v: isinstance(v, int) and v >= 0, self.window))

        self.window_op = get_window_op(window_method)

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:

        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        vid_qkv = gather_seq_scatter_heads_qkv(
            vid_qkv,
            seq_dim=0,
            qkv_shape=vid_shape,
            cache=cache.namespace("vid"),
        )
        txt_qkv = gather_seq_scatter_heads_qkv(
            txt_qkv,
            seq_dim=0,
            qkv_shape=txt_shape,
            cache=cache.namespace("txt"),
        )

        # re-org the input seq for window attn
        cache_win = cache.namespace(f"{self.window_method}_{self.window}_sd3")

        def make_window(x: torch.Tensor):
            t, h, w, _ = x.shape
            window_slices = self.window_op((t, h, w), self.window)
            return [x[st, sh, sw] for (st, sh, sw) in window_slices]

        window_partition, window_reverse, window_shape, window_count = cache_win(
            "win_transform",
            lambda: na.window_idx(vid_shape, make_window, fused_window_attn=getattr(self, "fused_window_attn", False)),
        )
        vid_qkv_win = window_partition(vid_qkv)

        vid_qkv_win = rearrange(vid_qkv_win, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv_win.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))

        vid_len_win = cache_win("vid_len", lambda: window_shape.prod(-1))
        txt_len_win = cache_win("txt_len", lambda: txt_len.repeat_interleave(window_count))
        all_len_win = cache_win("all_len", lambda: vid_len_win + txt_len_win)
        concat_win, unconcat_win = cache_win(
            "mm_pnp", lambda: na.repeat_concat_idx(vid_len_win, txt_len, window_count)
        )

        # window rope
        if self.rope:
            if self.rope.mm:
                # repeat text q and k for window mmrope
                _, num_h, _ = txt_q.shape
                txt_q_repeat = rearrange(txt_q, "l h d -> l (h d)")
                txt_q_repeat = na.unflatten(txt_q_repeat, txt_shape)
                txt_q_repeat = [[x] * n for x, n in zip(txt_q_repeat, window_count)]
                txt_q_repeat = list(chain(*txt_q_repeat))
                txt_q_repeat, txt_shape_repeat = na.flatten(txt_q_repeat)
                txt_q_repeat = rearrange(txt_q_repeat, "l (h d) -> l h d", h=num_h)

                txt_k_repeat = rearrange(txt_k, "l h d -> l (h d)")
                txt_k_repeat = na.unflatten(txt_k_repeat, txt_shape)
                txt_k_repeat = [[x] * n for x, n in zip(txt_k_repeat, window_count)]
                txt_k_repeat = list(chain(*txt_k_repeat))
                txt_k_repeat, _ = na.flatten(txt_k_repeat)
                txt_k_repeat = rearrange(txt_k_repeat, "l (h d) -> l h d", h=num_h)

                vid_q, vid_k, txt_q, txt_k = self.rope(
                    vid_q, vid_k, window_shape, txt_q_repeat, txt_k_repeat, txt_shape_repeat, cache_win
                )
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, window_shape, cache_win)
            
        # KV Sharing across tiles
        tile_coords = cache.get("tile_coords", None)
        kv_cache = cache.get("kv_cache", None)
        
        if tile_coords is not None and kv_cache is not None:
            iy, ix = tile_coords
            kv_namespace = cache.namespace("kv")
            block_idx = kv_namespace.get("block_idx", 0)
            kv_namespace.set("block_idx", block_idx + 1)
            
            B = vid_shape.shape[0]
            T, H, W = vid_shape[0].tolist()
            tt, hh, ww = window_shape[0].tolist()
            nt, nh, nw = T // tt, H // hh, W // ww
            
            window_size = tt * hh * ww
            _, h_dim, d_dim = vid_k.shape
            
            k_grid = vid_k.view(B, nt, nh, nw, window_size, h_dim, d_dim)
            v_grid = vid_v.view(B, nt, nh, nw, window_size, h_dim, d_dim)
            
            right_k = k_grid[:, :, :, nw-1:nw].clone()
            right_v = v_grid[:, :, :, nw-1:nw].clone()
            bottom_k = k_grid[:, :, nh-1:nh, :].clone()
            bottom_v = v_grid[:, :, nh-1:nh, :].clone()
            
            kv_cache[(block_idx, iy, ix)] = {
                'right_k': right_k, 'right_v': right_v,
                'bottom_k': bottom_k, 'bottom_v': bottom_v
            }
            
            left_dict = kv_cache.get((block_idx, iy, ix - 1)) if ix > 0 else None
            top_dict = kv_cache.get((block_idx, iy - 1, ix)) if iy > 0 else None
            
            left_k = left_dict['right_k'] if left_dict else None
            left_v = left_dict['right_v'] if left_dict else None
            top_k = top_dict['bottom_k'] if top_dict else None
            top_v = top_dict['bottom_v'] if top_dict else None
            
            if left_k is not None or top_k is not None:
                new_k_list, new_v_list, len_list = [], [], []
                for b in range(B):
                    for t in range(nt):
                        for y in range(nh):
                            for x in range(nw):
                                cur_k = k_grid[b, t, y, x]
                                cur_v = v_grid[b, t, y, x]
                                
                                to_cat_k, to_cat_v = [cur_k], [cur_v]
                                
                                if x == 0 and left_k is not None:
                                    to_cat_k.append(left_k[b, t, y, 0])
                                    to_cat_v.append(left_v[b, t, y, 0])
                                    
                                if y == 0 and top_k is not None:
                                    to_cat_k.append(top_k[b, t, 0, x])
                                    to_cat_v.append(top_v[b, t, 0, x])
                                    
                                cat_k = torch.cat(to_cat_k, dim=0)
                                cat_v = torch.cat(to_cat_v, dim=0)
                                
                                new_k_list.append(cat_k)
                                new_v_list.append(cat_v)
                                len_list.append(cat_k.shape[0])
                                
                vid_k = torch.cat(new_k_list, dim=0)
                vid_v = torch.cat(new_v_list, dim=0)
                vid_len_win_k = torch.tensor(len_list, dtype=torch.long, device=vid_k.device)
                
                concat_win_k, _ = cache_win(f"mm_pnp_k_{ix}_{iy}", lambda: na.repeat_concat_idx(vid_len_win_k, txt_len, window_count))
                all_len_win_k = vid_len_win_k + txt_len_win
            else:
                concat_win_k = concat_win
                all_len_win_k = all_len_win
        else:
            concat_win_k = concat_win
            all_len_win_k = all_len_win
            
        # Attention handles dtype conversion internally using pipeline compute_dtype
        out = self.attn(
            q=concat_win(vid_q, txt_q),
            k=concat_win_k(vid_k, txt_k),
            v=concat_win_k(vid_v, txt_v),
            cu_seqlens_q=cache_win(
                "vid_seqlens_q", lambda: safe_pad_operation(all_len_win.cumsum(0), (1, 0)).int()
            ),
            cu_seqlens_k=safe_pad_operation(all_len_win_k.cumsum(0), (1, 0)).int(),
            max_seqlen_q=cache_win("vid_max_seqlen_q", lambda: all_len_win.max()),
            max_seqlen_k=all_len_win_k.max(),
        ).type_as(vid_q)

        # text pooling
        vid_out, txt_out = unconcat_win(out)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")
        vid_out = window_reverse(vid_out)

        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)

        return vid_out, txt_out
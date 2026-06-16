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

from typing import Tuple, Union
import torch
from einops import rearrange
from torch.nn import functional as F

# from ..cache import Cache
from ....common.cache import Cache
from ....common.distributed.ops import gather_heads_scatter_seq, gather_seq_scatter_heads_qkv

from .. import na
from ..attention import FlashAttentionVarlen
from ..blocks.mmdit_window_block import MMWindowAttention, MMWindowTransformerBlock
from ..mm import MMArg
from ..modulation import ada_layer_type
from ..normalization import norm_layer_type
from ..rope import NaRotaryEmbedding3d
from ..window import get_window_op
from ....common.half_precision_fixes import safe_pad_operation

class NaSwinAttention(MMWindowAttention):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_rope: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        window: Union[int, Tuple[int, int, int]],
        window_method: str,
        shared_qkv: bool,
        attention_mode: str = 'sdpa',
        **kwargs,
    ):
        super().__init__(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_rope=qk_rope,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            window=window,
            window_method=window_method,
            shared_qkv=shared_qkv,
        )
        self.rope = NaRotaryEmbedding3d(dim=head_dim // 2) if qk_rope else None
        self.attn = FlashAttentionVarlen(attention_mode=attention_mode)
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


class NaMMSRTransformerBlock(MMWindowTransformerBlock):
    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: norm_layer_type,
        norm_eps: float,
        ada: ada_layer_type,
        qk_bias: bool,
        qk_rope: bool,
        qk_norm: norm_layer_type,
        shared_qkv: bool,
        shared_mlp: bool,
        mlp_type: str,
        **kwargs,
    ):
        super().__init__(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            emb_dim=emb_dim,
            heads=heads,
            head_dim=head_dim,
            expand_ratio=expand_ratio,
            norm=norm,
            norm_eps=norm_eps,
            ada=ada,
            qk_bias=qk_bias,
            qk_rope=qk_rope,
            qk_norm=qk_norm,
            shared_qkv=shared_qkv,
            shared_mlp=shared_mlp,
            mlp_type=mlp_type,
            **kwargs,
        )

        self.attn = NaSwinAttention(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_rope=qk_rope,
            qk_norm=qk_norm,
            qk_norm_eps=norm_eps,
            shared_qkv=shared_qkv,
            **kwargs,
        )

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        emb: torch.FloatTensor,
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
    ]:
        hid_len = MMArg(
            cache("vid_len", lambda: vid_shape.prod(-1)),
            cache("txt_len", lambda: txt_shape.prod(-1)),
        )
        ada_kwargs = {
            "emb": emb,
            "hid_len": hid_len,
            "cache": cache,
            "branch_tag": MMArg("vid", "txt"),
        }

        fused_adaln = getattr(self, "fused_adaln", False)
        
        if fused_adaln:
            try:
                from ....optimization.fused_adaln import fused_adaln_forward
                vid_res, txt_res = self.ada(vid, txt, layer="attn", mode="in_fused", **ada_kwargs)
                
                vid_norm_module = self.attn_norm.vid if not self.attn_norm.shared_weights else self.attn_norm.all
                vid_scaleA, vid_scaleB, vid_shiftA, vid_shiftB = vid_res
                vid_attn = fused_adaln_forward(vid, vid_scaleA, vid_shiftA, vid_scaleB, vid_shiftB, getattr(vid_norm_module, "eps", 1e-5))
                
                if not self.attn_norm.vid_only:
                    txt_norm_module = self.attn_norm.txt if not self.attn_norm.shared_weights else self.attn_norm.all
                    txt_scaleA, txt_scaleB, txt_shiftA, txt_shiftB = txt_res
                    txt_attn = fused_adaln_forward(txt, txt_scaleA, txt_shiftA, txt_scaleB, txt_shiftB, getattr(txt_norm_module, "eps", 1e-5))
                else:
                    txt_attn = txt
            except Exception as e:
                vid_attn, txt_attn = self.attn_norm(vid, txt)
                vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="in", **ada_kwargs)
        else:
            vid_attn, txt_attn = self.attn_norm(vid, txt)
            vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="in", **ada_kwargs)
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, vid_shape, txt_shape, cache)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="out", **ada_kwargs)
        vid_attn, txt_attn = (vid_attn + vid), (txt_attn + txt)
        del vid, txt  # <--- explicitly free the input tensors early!

        if fused_adaln:
            try:
                from ....optimization.fused_adaln import fused_adaln_forward
                vid_res, txt_res = self.ada(vid_attn, txt_attn, layer="mlp", mode="in_fused", **ada_kwargs)
                
                vid_norm_module = self.mlp_norm.vid if not self.mlp_norm.shared_weights else self.mlp_norm.all
                vid_scaleA, vid_scaleB, vid_shiftA, vid_shiftB = vid_res
                vid_mlp = fused_adaln_forward(vid_attn, vid_scaleA, vid_shiftA, vid_scaleB, vid_shiftB, getattr(vid_norm_module, "eps", 1e-5))
                
                if not self.mlp_norm.vid_only:
                    txt_norm_module = self.mlp_norm.txt if not self.mlp_norm.shared_weights else self.mlp_norm.all
                    txt_scaleA, txt_scaleB, txt_shiftA, txt_shiftB = txt_res
                    txt_mlp = fused_adaln_forward(txt_attn, txt_scaleA, txt_shiftA, txt_scaleB, txt_shiftB, getattr(txt_norm_module, "eps", 1e-5))
                else:
                    txt_mlp = txt_attn
                    
                if vid_mlp.dtype != vid_attn.dtype:
                    vid_mlp = vid_mlp.to(vid_attn.dtype)
                if not self.mlp_norm.vid_only and txt_mlp.dtype != txt_attn.dtype:
                    txt_mlp = txt_mlp.to(txt_attn.dtype)
            except Exception as e:
                vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
                if vid_mlp.dtype != vid_attn.dtype:
                    vid_mlp = vid_mlp.to(vid_attn.dtype)
                if txt_mlp.dtype != txt_attn.dtype:
                    txt_mlp = txt_mlp.to(txt_attn.dtype)
                vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="in", **ada_kwargs)
        else:
            vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
            if vid_mlp.dtype != vid_attn.dtype:
                vid_mlp = vid_mlp.to(vid_attn.dtype)
            if txt_mlp.dtype != txt_attn.dtype:
                txt_mlp = txt_mlp.to(txt_attn.dtype)
            vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="in", **ada_kwargs)
        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="out", **ada_kwargs)
        vid_mlp, txt_mlp = (vid_mlp + vid_attn), (txt_mlp + txt_attn)
        del vid_attn, txt_attn  # <--- explicitly free intermediate tensors early!

        return vid_mlp, txt_mlp, vid_shape, txt_shape

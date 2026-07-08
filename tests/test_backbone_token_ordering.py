"""
Regression test for the divided space-time token ordering in
backbone/vision_transformer.py.

Background
----------
`VisionTransformer.forward_features` builds the patch-token sequence frame-major:
    x = patch_embed(x)         # (B, T, N, D),  N = H*W patches per frame
    x = x.reshape(B, T*N, D)   # flat position = t*N + n   -> layout "(t h w)"

Every `Block` must therefore interpret that sequence as "(t h w)" when it
splits it into temporal groups (one spatial location across all frames) and
spatial groups (all patches within one frame). An earlier version used the
"(h w t)" pattern, which silently scrambled space and time.

This test tags each token with its (frame_index, patch_index) identity and
verifies that the rearrange patterns used by the Block recover the correct
grouping. Run with:  python tests/test_backbone_token_ordering.py
"""
import torch
from einops import rearrange


def _make_tagged_sequence(B, T, H, W, D=2):
    """Return (B, T*H*W, D) where channel 0 = frame index, channel 1 = patch index,
    laid out exactly as forward_features lays it out (frame-major, '(t h w)')."""
    N = H * W
    seq = torch.zeros(B, T * N, D)
    for t in range(T):
        for n in range(N):
            pos = t * N + n          # forward_features: reshape(B, T*N, D)
            seq[:, pos, 0] = t
            seq[:, pos, 1] = n
    return seq


def test_temporal_grouping():
    B, T, H, W = 2, 4, 3, 5
    seq = _make_tagged_sequence(B, T, H, W)
    # Pattern used by Block for temporal attention:
    xt = rearrange(seq, 'b (t h w) d -> (b h w) t d', b=B, h=H, w=W, t=T)
    # Each temporal group must be ONE patch (constant patch idx) across ALL frames 0..T-1
    for g in range(xt.shape[0]):
        frames = xt[g, :, 0].tolist()
        patches = xt[g, :, 1].tolist()
        assert frames == list(range(T)), f"group {g}: frames not 0..T-1 in order -> {frames}"
        assert len(set(patches)) == 1, f"group {g}: patch idx not constant over time -> {patches}"
    print("[OK] temporal groups = one spatial location across all frames")


def test_spatial_grouping():
    B, T, H, W = 2, 4, 3, 5
    N = H * W
    seq = _make_tagged_sequence(B, T, H, W)
    # Pattern used by Block for spatial attention:
    xs = rearrange(seq, 'b (t h w) d -> (b t) (h w) d', b=B, h=H, w=W, t=T)
    # Each spatial group must be ONE frame (constant frame idx) covering all patches 0..N-1
    for g in range(xs.shape[0]):
        frames = xs[g, :, 0].tolist()
        patches = xs[g, :, 1].tolist()
        assert len(set(frames)) == 1, f"group {g}: frame idx not constant -> {frames}"
        assert sorted(patches) == list(range(N)), f"group {g}: patches don't cover 0..N-1 -> {patches}"
    print("[OK] spatial groups = all patches within one frame")


def test_roundtrip_identity():
    """The temporal/spatial rearrange + inverse must return the sequence unchanged."""
    B, T, H, W = 2, 4, 3, 5
    seq = _make_tagged_sequence(B, T, H, W)
    xt = rearrange(seq, 'b (t h w) d -> (b h w) t d', b=B, h=H, w=W, t=T)
    back = rearrange(xt, '(b h w) t d -> b (t h w) d', b=B, h=H, w=W, t=T)
    assert torch.equal(seq, back), "temporal rearrange round-trip changed the sequence"
    xs = rearrange(seq, 'b (t h w) d -> (b t) (h w) d', b=B, h=H, w=W, t=T)
    back2 = rearrange(xs, '(b t) (h w) d -> b (t h w) d', b=B, h=H, w=W, t=T)
    assert torch.equal(seq, back2), "spatial rearrange round-trip changed the sequence"
    print("[OK] rearrange round-trips are identity")


if __name__ == "__main__":
    test_temporal_grouping()
    test_spatial_grouping()
    test_roundtrip_identity()
    print("\nAll token-ordering checks passed.")

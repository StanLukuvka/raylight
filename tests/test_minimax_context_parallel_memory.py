from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "src/raylight/diffusion_models/minimax/xdit_context_parallel.py"
)


def test_distributed_attention_preserves_native_inplace_rms_rope():
    source = SOURCE.read_text()
    attention = source[source.index("def usp_attn_forward"):source.index("def usp_dit_forward")]
    assert "comfy.quant_ops.ck.rms_rope_split_half_(" in attention
    assert "qw = comfy.model_management.cast_to(self.q_norm.weight" in attention
    assert "kw = comfy.model_management.cast_to(self.k_norm.weight" in attention
    assert "q = self.q_norm(q)" not in attention
    assert "k = self.k_norm(k)" not in attention
    assert "apply_rope_split_half(" not in attention


def test_sequence_parallel_split_detaches_local_storage():
    source = SOURCE.read_text()
    assert "local_h = h[start:end].clone()" in source
    assert "local_rope = rope_freqs[:, start:end].clone()" in source
    assert "return local_h, local_rope, local_segments" in source


def test_full_sequence_embeddings_are_released_before_blocks():
    source = SOURCE.read_text()
    split = source.index("_split_packed_sequence(h, rope_freqs, mod_segments)")
    blocks = source.index("# blocks", split)
    release = source.index("del video_embed", split)
    assert split < release < blocks
    for name in (
        "video_embed",
        "audio_embed",
        "all_video_rows",
        "all_audio_rows",
        "video_rows",
        "audio_rows",
    ):
        assert name in source[release:blocks]

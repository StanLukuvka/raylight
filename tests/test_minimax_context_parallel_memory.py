from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "src/raylight/diffusion_models/minimax/xdit_context_parallel.py"
)


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

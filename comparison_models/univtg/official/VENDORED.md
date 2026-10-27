# Vendored: UniVTG (official code, unmodified)

Source: https://github.com/showlab/UniVTG, commit `32659ac7aeba21742a63274f30eba785fc57e247` (2024-05-08).
Licence: MIT (see `LICENSE` in this folder, copied from the repository).
Paper: K. Q. Lin et al., "UniVTG: Towards Unified Video-Language Temporal Grounding", ICCV 2023.

Only the files the model and its losses need are copied, byte for byte. The two `__init__.py` files
(one comment line each) are added so the folders import as packages; nothing else is added or changed.

| file | sha256 |
|---|---|
| model/univtg.py | 4451c4f0e9f8ca4c088fcd5251c20e8245aa162cca367e6bf6d684c1fe918297 |
| model/transformer_encoder_droppath.py | 399501234b0f457b1af3a19249bbc5983a0b5dab1d8d8886f124ed6d26acda36 |
| model/position_encoding.py | 64995b7501c6b226afe836d556ae90192402196b692284ae9554e0e9387fcfe8 |
| model/matcher.py | 40b8c314b52612aa13e6a7156d1e644762a2dfa2631dd214b497d7d3314e197c |
| utils/span_utils.py | e1eed63f9062730ac00d030484cdeb7fcc31c1ba17c0c670bb761fe3981b33e2 |
| utils/temporal_nms.py | 5793a161f99692cd4ede98634da153eb532b9dad4b261facd75b9b9566da7b59 |
| LICENSE | cd359473e1f1047b0b89d43ca657171d077ee56ac9aff7491111dfff8fb04ab5 |

`../adapter.py` feeds our data to this model (features, targets, training loop and window decoding
re-implemented from the repository's `main/dataset.py`, `main/train_mr.py` and `main/inference_mr.py`);
deviations from the original recipe, and why, are listed at the top of `../adapter.py`.

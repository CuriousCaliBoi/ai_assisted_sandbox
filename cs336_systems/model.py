from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM

from cs336_systems.flash_attention import patch_model_flash_attention


class FlashBasicsTransformerLM(BasicsTransformerLM):
    """BasicsTransformerLM with FA4 CuTe attention and optional layer checkpointing."""

    def __init__(self, *args, checkpoint_every: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        patch_model_flash_attention(self)
        self.checkpoint_every = checkpoint_every

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_hidden(x)
        return self.lm_head(x)

    def forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        embedded_tokens = self.token_embeddings(x)
        x = embedded_tokens

        if self.checkpoint_every <= 0:
            for layer in self.layers:
                x = layer(x)
        else:
            for start in range(0, len(self.layers), self.checkpoint_every):
                group = self.layers[start : start + self.checkpoint_every]

                def run_group(hidden: torch.Tensor, group=group) -> torch.Tensor:
                    for layer in group:
                        hidden = layer(hidden)
                    return hidden

                x = checkpoint(run_group, x, use_reentrant=False)

        return self.ln_final(x)


def build_flash_basics_transformer_lm(*, checkpoint_every: int = 0, **kwargs) -> FlashBasicsTransformerLM:
    return FlashBasicsTransformerLM(**kwargs, checkpoint_every=checkpoint_every)

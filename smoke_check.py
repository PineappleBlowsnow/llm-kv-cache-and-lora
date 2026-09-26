"""Small independent CPU checks added for the public portfolio snapshot.

No datasets, checkpoints, network access, or classroom grader are required.
Run from the repository root: python smoke_check.py
"""

import torch
from torch.nn import functional as F

from kv_cache.kv_cache import GPT_kv
from lora.lora import GPT_LoRA, LoRALinear, get_lora_model
from mingpt.model import GPT
from mingpt.utils import CfgNode


def config():
    cfg = GPT.get_default_config()
    cfg.model_type = None
    cfg.n_layer, cfg.n_head, cfg.n_embd = 2, 2, 16
    cfg.vocab_size, cfg.block_size = 7, 8
    cfg.embd_pdrop = cfg.resid_pdrop = cfg.attn_pdrop = 0.0
    return cfg


def check_cached_decoding():
    base, cached = GPT(config()).eval(), GPT_kv(config()).eval()
    cached.load_state_dict(base.state_dict())
    tokens = torch.randint(0, 7, (2, 5))
    with torch.no_grad():
        reference, _ = base(tokens)
        _, _, state = cached(tokens[:, :3], kv_cache=[None] * 2, compute_first=True)
        suffix, _, state = cached(tokens, kv_cache=state)
        torch.testing.assert_close(suffix, reference[:, 3:], atol=1e-6, rtol=1e-5)
        assert all(layer[0].shape[1] == 5 for layer in state)
        # Extends beyond block_size to exercise cache rebuilding for a sliding window.
        torch.testing.assert_close(
            cached.generate_kv(tokens, 6, do_sample=False),
            base.generate(tokens, 6, do_sample=False),
        )
    print("PASS: cached suffix logits, cache length, greedy decoding across context overflow")


def check_lora_update():
    layer = LoRALinear(5, 4, lora_rank=2, lora_alpha=4)
    with torch.no_grad():
        layer.lora_B.normal_(std=0.1)
    x = torch.randn(3, 5)
    before = layer.weight.detach().clone()
    expected = F.linear(x, before, layer.bias) + layer.lora_scaling * F.linear(
        F.linear(x, layer.lora_A), layer.lora_B
    )
    torch.testing.assert_close(layer(x), expected)
    layer.eval()  # Explicit layer-level merge API in the original implementation.
    assert layer.has_weights_merged
    torch.testing.assert_close(layer(x), expected)
    layer.train()
    assert not layer.has_weights_merged
    torch.testing.assert_close(layer.weight, before)

    cfg = config()
    cfg.lora_rank, cfg.lora_alpha = 2, 4
    model = get_lora_model(GPT_LoRA(cfg))
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    assert all("lora" in name for name, p in model.named_parameters() if p.requires_grad)
    optimizer = model.configure_optimizers(CfgNode(learning_rate=1e-3, betas=(0.9, 0.95), weight_decay=0.0))
    tokens = torch.randint(0, 7, (2, 5))
    _, loss = model(tokens, targets=tokens)
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for name, p in model.named_parameters() if "lora_B" in name)
    optimizer.step()
    for name, param in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(param, frozen[name], rtol=0, atol=0)
    print("PASS: low-rank formula, layer merge/unmerge, adapter gradients, frozen base after one step")


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(1)
    check_cached_decoding()
    check_lora_update()
    print("All public smoke checks passed.")

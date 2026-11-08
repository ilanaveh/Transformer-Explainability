"""
15/09/2026
Make sure that PoolingBlock and regular LRP Block return the same output when pool_config.keep_rate = 1.0.
"""

import torch
from baselines.ViT.ViT_LRP import Block, PoolingBlock


def sanity_check_pooling_block(
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=2,
    num_tokens=161,   # CLS + 160 patches, matching your APViT input
    dim=768,
    num_heads=12,
    mlp_ratio=4.0,
):
    """
    Compare a normal Transformer Block against a PoolingBlock
    configured with keep_rate=1.0.

    Since no tokens should be removed, the two outputs should be
    numerically identical (up to floating-point precision), assuming
    the forward implementations are equivalent.
    """

    torch.manual_seed(0)

    # ---------------------------------------------------------
    # 1. Create normal block
    # ---------------------------------------------------------
    normal_block = Block(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
    ).to(device)

    # ---------------------------------------------------------
    # 2. Create pooling block, but disable actual pruning
    #    by using keep_rate=1.0
    # ---------------------------------------------------------
    pooling_block = PoolingBlock(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        pool_config=dict(
            keep_rate=1.0,
            exclude_first=True,
            attn_method="SUM",
        ),
    ).to(device)

    # ---------------------------------------------------------
    # 3. Copy EXACTLY the same parameters
    # ---------------------------------------------------------
    load_result = pooling_block.load_state_dict(
        normal_block.state_dict(),
        strict=True
    )

    print("Weight loading:")
    print(load_result)

    # ---------------------------------------------------------
    # 4. Evaluation mode
    # ---------------------------------------------------------
    normal_block.eval()
    pooling_block.eval()

    # ---------------------------------------------------------
    # 5. Same input to both
    # ---------------------------------------------------------
    x = torch.randn(
        batch_size,
        num_tokens,
        dim,
        device=device
    )

    # clone() is useful just to guarantee the two calls don't
    # accidentally share some modified tensor.
    x1 = x.clone()
    x2 = x.clone()

    # Do NOT use torch.no_grad() here.
    #
    # Your Attention code calls:
    #     attn.register_hook(...)
    #
    # which requires autograd to be enabled.
    y_normal = normal_block(x1)
    y_pooling = pooling_block(x2)

    # ---------------------------------------------------------
    # 6. Compare
    # ---------------------------------------------------------
    diff = (y_normal - y_pooling).abs()

    print("\nOutput shapes:")
    print("normal :", y_normal.shape)
    print("pooling:", y_pooling.shape)

    print("\nOutput difference:")
    print("max abs diff :", diff.max().item())
    print("mean abs diff:", diff.mean().item())

    print(
        "allclose (1e-5):",
        torch.allclose(
            y_normal,
            y_pooling,
            atol=1e-5,
            rtol=1e-5
        )
    )

    # ---------------------------------------------------------
    # 7. Also compare the actual attention matrices
    # ---------------------------------------------------------
    attn_normal = normal_block.attn.get_attn()
    attn_pooling = pooling_block.attn.get_attn()

    attn_diff = (attn_normal - attn_pooling).abs()

    print("\nAttention shapes:")
    print("normal :", attn_normal.shape)
    print("pooling:", attn_pooling.shape)

    print("\nAttention difference:")
    print("max abs diff :", attn_diff.max().item())
    print("mean abs diff:", attn_diff.mean().item())

    print(
        "attention allclose (1e-5):",
        torch.allclose(
            attn_normal,
            attn_pooling,
            atol=1e-5,
            rtol=1e-5
        )
    )

    # ---------------------------------------------------------
    # 8. Verify that keep_rate=1 really performed no pooling
    # ---------------------------------------------------------
    print("\nPooling:")
    print(
        "pool_keep_index:",
        pooling_block.pool_keep_index
    )

    if pooling_block.pool_keep_index is None:
        print("PASS: no tokens were pruned.")
    else:
        print(
            "WARNING: tokens were unexpectedly pruned:",
            pooling_block.pool_keep_index.shape
        )

    return {
        "max_output_diff": diff.max().item(),
        "mean_output_diff": diff.mean().item(),
        "max_attention_diff": attn_diff.max().item(),
        "mean_attention_diff": attn_diff.mean().item(),
    }


results = sanity_check_pooling_block()
print()

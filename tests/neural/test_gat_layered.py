import typing as T

import torch
from torch import nn
from tqdm import tqdm  # type: ignore

from Gat.neural import layers
from tests.conftest import GatSetup


def test_gat_layered_converge(gat_setup: GatSetup, device: torch.device) -> None:

    if not device == torch.device("cuda"):
        lsnum_heads = [4, 12]
    else:
        lsnum_heads = [2, 4, 12, 64, 384]

    crs_entrpy = nn.CrossEntropyLoss()
    n_steps = 5000

    steps_to_converge_for_num_heads: T.Dict[int, int] = {}
    for num_heads in lsnum_heads:
        head_size = gat_setup.all_config.model.embed_dim // num_heads
        # Edge features are dependent on head size
        key_edge_features: torch.FloatTensor = torch.randn(  # type: ignore
            gat_setup.all_config.trainer.train_batch_size,
            gat_setup.seq_length,
            gat_setup.seq_length,
            head_size,
            names=("B", "N_left", "N_right", "head_size"),
            requires_grad=True,
            device=device,
        )

        model_config = gat_setup.all_config.model
        gat_layered = layers.GATLayered(
            num_heads=model_config.num_heads,
            edge_dropout_p=model_config.num_heads,
            rezero_or_residual=model_config.rezero_or_residual,
            intermediate_dim=model_config.intermediate_dim,
            num_layers=model_config.num_layers,
            feat_dropout_p=model_config.feat_dropout_p,
            lsnode_feature_embedder=lsnode_feature_embedder,
            key_edge_feature_embedder=key_edge_feature_embedder,
        )

        gat_layered.to(device)
        adam = torch.optim.Adam(
            # Include key_edge_features in features to be optimized
            [key_edge_features]
            + T.cast(T.List[torch.FloatTensor], list(gat_layered.parameters())),
            lr=1e-3,
        )
        gat_layered.train()
        for step in tqdm(range(1, n_steps + 1), desc=f"num_heads={num_heads}"):
            after_self_att = gat_layered(
                adj=gat_setup.adj,
                node_ids=gat_setup.node_features,
                key_edge_features=key_edge_features,
            )
            loss = crs_entrpy(
                after_self_att.flatten(["B", "N"], "BN").rename(None),
                gat_setup.node_labels.flatten(["B", "N"], "BN").rename(None),
            )
            preds = after_self_att.rename(None).argmax(dim=-1)
            if torch.all(torch.eq(preds, gat_setup.node_labels)):
                steps_to_converge_for_num_heads[num_heads] = step
                break
            loss.backward()
            adam.step()
        if not torch.all(torch.eq(preds, gat_setup.node_labels)):
            print(f"Did not converge for num_heads={num_heads}")
    print(
        "converged for heads: "
        + " | ".join(
            f"num_heads: {num_heads}, step: {steps}"
            for num_heads, steps in steps_to_converge_for_num_heads.items()
        )
    )

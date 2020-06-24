import torch

from Gat import config


class TestMixin:
    def setUp(self) -> None:
        super().setUp()  # type: ignore
        # There's no super class now, there will be, since this is amixin.


class DeviceMixin(TestMixin):
    def setUp(self) -> None:
        # This was stolen from private torch code
        if hasattr(torch._C, "_cuda_isDriverSufficient") and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        super().setUp()


class EverythingConfigMixin(TestMixin):
    """

    Sets:
        self._all_config
    """

    def setUp(self) -> None:

        self._all_config = config.EverythingConfig(
            config.TrainerConfig(
                lr=1e-3,
                epochs=99,
                train_batch_size=3,
                eval_batch_size=2,
                use_cuda=False,
            ),
            preprop=config.PreprocessingConfig(
                undirected=True,
                dataset_dir="actual_data/SST-2_tiny",
                sent2graph_name="dep",
            ),
            model=config.GATForSequenceClassificationConfig(
                config.GATLayeredConfig(num_heads=1, intermediate_dim=20, num_layers=3),
                embedding_dim=768,
                node_embedding_type="basic",
                use_edge_features=True,
                dataset_dep=None,
            ),
        )
        super().setUp()

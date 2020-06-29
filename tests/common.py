import torch

from Gat import configs
from Gat import testing_utils


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

    @testing_utils.debug_on()
    def setUp(self) -> None:

        self._all_config = configs.EverythingConfig(
            configs.TrainerConfig(
                lr=1e-3,
                epochs=99,
                train_batch_size=3,
                eval_batch_size=2,
                use_cuda=False,
            ),
            preprop=configs.PreprocessingConfig(
                undirected=True,
                dataset_dir="actual_data/SST-2_tiny",
                sent2graph_name="dep",
                unk_thres=None,
            ),
            model=configs.GATForSequenceClassificationConfig(
                configs.GATLayeredConfig(
                    num_heads=2, intermediate_dim=20, num_layers=3
                ),
                use_pretrained_embs=False,
                embedding_dim=300,
                node_embedding_type="bpe",
                bpe_vocab_size=25000,
                use_edge_features=True,
                dataset_dep=None,
            ),
        )
        super().setUp()

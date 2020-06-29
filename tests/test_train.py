import typing as T
import unittest.mock
import warnings

import torch
from pytorch_lightning.logging import TensorBoardLogger  # type: ignore
from pytorch_lightning.trainer import Trainer  # type: ignore

import train
from Gat import testing_utils
from tests.common import EverythingConfigMixin

warnings.simplefilter("ignore")


class LitGatForSequenceClassificationMixin(EverythingConfigMixin):
    def setUp(self) -> None:
        super().setUp()
        self._lit_model = train.LitGatForSequenceClassification(self._all_config)

    def _get_params(self) -> T.Dict[str, T.List[torch.Tensor]]:
        return dict(
            [
                (module.debug_name, [param.clone() for param in module.parameters()])  # type: ignore
                for module in self._lit_model._gat_model.modules()
                if hasattr(module, "debug_name")
            ]
        )

    def _do_training(self) -> None:
        trainer = Trainer(max_epochs=1, logger=TensorBoardLogger("./tb_logs"))
        trainer.fit(self._lit_model)


class TestItRuns(LitGatForSequenceClassificationMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    @testing_utils.debug_on()
    def test_it(self) -> None:
        self._do_training()

    def tearDown(self) -> None:
        super().tearDown()


@unittest.skip("not sure if I know which things should be updated yet, cuz of rezero.")
class TestBackprop(LitGatForSequenceClassificationMixin, unittest.TestCase):
    @testing_utils.debug_on()
    def test_node_embeddings_updated(self) -> None:
        class MockLitGatForSequenceClassification(
            train.LitGatForSequenceClassification
        ):
            def on_train_start(self) -> None:
                # Add to vocab a node that doens't exist in data
                super().on_train_start()
                print("Mocked on train start called!!!")
                self._word_vocab._id2word.append("DOESNT_EXIST_IN_DATA")
                new_tok_id = len(self._word_vocab._id2word) - 1
                assert new_tok_id not in self._word_vocab._word2id.values()
                self._word_vocab._word2id["DOESNT_EXIST_IN_DATA"] = new_tok_id

        with unittest.mock.patch(
            "train.LitGatForSequenceClassification", MockLitGatForSequenceClassification
        ):
            self._lit_model = train.LitGatForSequenceClassification(self._all_config)
            self._lit_model.setup("fit")
            # params_before = self._get_params()
            self._do_training()
            # params_after = self._get_params()
            breakpoint()

    @testing_utils.debug_on()
    def test_right_things_updated(self) -> None:
        self.skipTest("need to update to use new TestCase structure.")
        # Build data and model
        self._lit_model._setup_data()
        self._lit_model._setup_model()

        all_edge_ids = set(
            range(len(self._lit_model._datasets["train"].edge_type2id.values()))
        )
        # Get the edges that are not in the dataset.
        # Begin with all edges.
        edge_ids_not_used_in_train_dataset = all_edge_ids.copy()

        for example in self._lit_model._datasets["train"]:
            edge_ids_not_used_in_train_dataset -= set(example.lsgraph[0].lsedge_type)

        if not edge_ids_not_used_in_train_dataset:
            self.skipTest("All edges found in training dataset.")
        if all_edge_ids == edge_ids_not_used_in_train_dataset:
            self.skipTest("No edges in the training dataset.")

        edge_embedder = (
            self._lit_model._gat_model._gat_layered._key_edge_feature_embedder
        )

        # It's not documented that this could *not* be the positional embedder,
        # but we know that it's  not since we wrote the code.
        assert edge_embedder is not None

        unused_tok_ids: torch.LongTensor = torch.tensor(  # type: ignore
            list(edge_ids_not_used_in_train_dataset)
        )
        used_tok_ids: torch.LongTensor = torch.tensor(  # type: ignore
            list(all_edge_ids - edge_ids_not_used_in_train_dataset)
        )

        unused_before_training = edge_embedder(unused_tok_ids).clone().detach()
        used_before_training = edge_embedder(used_tok_ids).clone().detach()

        unused_after_training = edge_embedder(unused_tok_ids)
        used_after_training = edge_embedder(used_tok_ids)

        print("not used edge ids:", edge_ids_not_used_in_train_dataset)
        print("used edge ids:", all_edge_ids - edge_ids_not_used_in_train_dataset)

        breakpoint()
        with self.subTest("unused_edge_ids"):
            self.assertTrue(
                (unused_before_training == unused_after_training).rename(None).all()
            )
        with self.subTest("used_edge_ids"):
            breakpoint()
            self.assertTrue(
                (used_before_training != used_after_training).rename(None).any()
            )


if __name__ == "__main__":
    unittest.main()

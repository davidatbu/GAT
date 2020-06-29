import logging
import typing as T
from pathlib import Path

import typing_extensions as TT

from Gat.data.datasets import BaseSentenceToGraphDataset
from Gat.data.datasets import SentenceGraphDataset
from Gat.data.sent2graphs import DepSentenceToGraph
from Gat.data.sent2graphs import SentenceToGraph
from Gat.data.sent2graphs import SRLSentenceToGraph
from Gat.data.sources import CsvTextSource
from Gat.data.tokenizers import WrappedSpacyTokenizer
from Gat.data.vocabs import BasicVocab

logger = logging.getLogger(__name__)
SENT2GRAPHS: T.Dict[str, T.Type[SentenceToGraph]] = {
    "srl": SRLSentenceToGraph,
    "dep": DepSentenceToGraph,
}


def load_splits(
    dataset_dir: Path,
    sent2graph_name: TT.Literal["srl", "dep"],
    splits: T.List[str] = ["train", "val"],
    fp_ending: str = "tsv",
    lstxt_col: T.List[str] = ["sentence"],
    lbl_col: str = "label",
    delimiter: str = "\t",
    unk_thres: T.Optional[int] = 1,
) -> T.Tuple[
    T.Dict[str, BaseSentenceToGraphDataset[BasicVocab]],
    T.Dict[str, CsvTextSource],
    BasicVocab,
]:
    """Build `Vocab` and `Labels` from training data. Process all splits."""
    assert "train" in splits
    txt_srcs = {
        split: CsvTextSource(
            fp=(dataset_dir / f"{split}.{fp_ending}"),
            lstxt_col=lstxt_col,
            lbl_col=lbl_col,
            csv_reader_kwargs={"delimiter": delimiter},
        )
        for split in splits
    }

    vocab = BasicVocab(
        txt_src=txt_srcs["train"],
        cache_dir=dataset_dir,
        unk_thres=unk_thres,
        tokenizer=WrappedSpacyTokenizer(),
    )

    cls_sent2graph = SENT2GRAPHS[sent2graph_name]
    split_datasets: T.Dict[str, BaseSentenceToGraphDataset[BasicVocab]] = {
        split: SentenceGraphDataset(
            cache_dir=dataset_dir,
            txt_src=txt_src,
            sent2graph=cls_sent2graph(),
            vocab=vocab,
            processing_batch_size=1000,
        )
        for split, txt_src in txt_srcs.items()
    }

    logger.info("First 10 of each split")
    for split, dataset in split_datasets.items():
        logger.info(f"{split}")
        for i in range(min(len(dataset), 5)):
            lssentgraph, lbl_id = dataset[i]
            logger.info(f"{vocab.labels.get_lbl(lbl_id)}")
            for _, _, _, nodeid2wordid in lssentgraph:
                assert nodeid2wordid is not None
                logger.info(f"\t{vocab.get_lstok(nodeid2wordid)}")

    return split_datasets, txt_srcs, vocab

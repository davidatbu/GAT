"""Currently, this exists for the sole purpose of pooling over BERT's subword tokens."""
import itertools
import typing as T

import torch
from torch import nn

from Gat import data
from Gat import neural


if T.TYPE_CHECKING:
    nnModule = nn.Module[torch.Tensor]
else:
    nnModule = nn.Module


class TokenizingReconciler(nnModule):
    def __init__(
        self,
        sub_word_vocab: data.Vocab,
        word_tokenizer: data.tokenizers.Tokenizer,
        sub_word_embedder: neural.layers.Embedder,
    ) -> None:
        """Pool over subword embeddings.

        Args:
            sub_word_vocab: We access `sub_word_vocab.tokenizer` and
                `sub_word_vocab.padding_tok_id`.
            word_tokenizer:
            sub_word_embedder: We use it to get subword embeddings, and access
                `sub_.wrod_embedder.max_seq_len`.
        """
        super().__init__()
        self._sub_word_vocab = sub_word_vocab
        self._word_tokenizer = word_tokenizer
        self._sub_word_embedder = sub_word_embedder

    def forward(self, lstxt: T.List[str]) -> torch.Tensor:
        """Tokenize using the two tokenizers, pool over subwords to create word embedding.

        Args:
            lstxt: A list of sentences.
        Returns:
            embedding: (B, L, E)
                       B = len(lstxt)
                       L is computed like this:
                       The sentences are tokenized by self.word_vocab.tokenizer, and
                       truncated to the last word whose complete sub word tokenization
                       "fits inside" the maximum number of sub word tokens allowed by
                       `sub_word_embedder` per sequence.
                       L is the number of words in the sentence with the most word
                       tokens after the truncation described above.

                       For example,

                       sent = "love embeddings"
                       sub_word_tokenization = [ "love", "embed", "#dings" ]
                       word_tokenization = [ "love", "embeddings" ]
                       sub_word_embedder.max_seq_len == 2 # True
                       L == 1 # True, since "embeddings" doesn't fit within 2 sub word
                              # tokens
        """
        lswords = self._word_tokenizer.batch_tokenize(lstxt)
        lssubwordids_per_word = [
            [self._sub_word_vocab.tokenize_and_get_lstok_id(word) for word in words]
            for words in lswords
        ]
        # "Flat" sub word tokenization for each sequence
        lslssubwordid: T.List[T.List[int]] = []
        # The number of sub words in each word
        lssubword_counts: T.List[T.List[int]] = []

        max_subword_seq_len = float("inf")
        if self._sub_word_embedder.max_seq_len is not None:
            max_subword_seq_len = float(self._sub_word_embedder.max_seq_len)
        for subwordids_per_word in lssubwordids_per_word:
            # "Flatten" to one list
            subwordids: T.List[int] = []
            subword_counts: T.List[int] = []
            for subwordids_for_one_word in subwordids_per_word:
                # Check if subword tokenization exceeds the limit
                if len(subwordids) > max_subword_seq_len:
                    break
                subwordids.extend(subwordids_for_one_word)
                subword_counts.append(len(subwordids_for_one_word))
            lssubword_counts.append(subword_counts)
            lslssubwordid.append(subwordids)

        subword_ids = torch.tensor(
            lslssubwordid, dtype=torch.long, device=next(self.parameters()).device
        )
        subword_embs = self._sub_word_embedder(subword_ids)

        pooled_word_embs = self.pool_sequences(subword_embs, lssubword_counts)
        return pooled_word_embs

    def pool_sequences(
        self, subword_seqs: torch.Tensor, lssubword_counts: T.List[T.List[int]]
    ) -> torch.Tensor:
        """Pool over sub word embeddings to yield word embeddings.

        Args:
            subword_seqs: (B, L, ...)
            lssubword_counts: The number of subwords within each "word".

        Returns:
            word_seqs: (B, L, ...)
                L here will be max([ sum(subword_counts) for subword_counts in
                lssubword_counts ])
        """
        # Check sub word sequences lengths fit within subword_seq.shape
        max_subword_seq_len = max(
            [sum(subword_counts) for subword_counts in lssubword_counts]
        )
        assert max_subword_seq_len <= subword_seqs.size("L")

        # Figure out the longest word seq length
        max_word_seq_len = max(map(len, lssubword_counts))

        # Get the padding vector
        padding_vec = self._sub_word_embedder(
            torch.tensor(
                [[self._sub_word_vocab.get_tok_id(self._sub_word_vocab.padding_tok)]]
            )
        )
        padding_vec = padding_vec.squeeze()

        # Word embeddings per seq
        lsword_seq: T.List[torch.Tensor] = []

        for subword_seq, subword_counts in zip(subword_seqs, lssubword_counts):
            beg_and_end_indices = itertools.accumulate([0] + subword_counts)
            beg_iterator, end_iterator = itertools.tee(beg_and_end_indices, 2)

            next(end_iterator)  # Consume the 0 at the beginning
            word_seq_len = len(subword_counts)
            word_seq = torch.stack(
                [
                    subword_seq[beg:end].mean(dim=0).rename(None)
                    for beg, end in zip(beg_iterator, end_iterator)
                ]
                + [padding_vec] * (max_word_seq_len - word_seq_len)
            )

            # TODO: Remove
            assert len(word_seq) == len(subword_counts)
            lsword_seq.append(word_seq)
        word_seqs = torch.stack(lsword_seq).rename("B", "L", "E")  # type: ignore
        return word_seqs

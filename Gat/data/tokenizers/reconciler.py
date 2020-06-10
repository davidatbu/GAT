import typing as T

import torch

from Gat import data  # TODO: Inconcistent intra-package import convention
from Gat import neural


class TokenizingReconciler:
    def __init__(
        self,
        sub_word_vocab: data.Vocab,
        word_vocab: data.Vocab,
        sub_word_embedder: neural.layers.Embedder,
        word_embedder: neural.layers.Embedder,
        sub_word_max_tokens_per_seq: int,
    ) -> None:
        """Given
            1. two tokenizations (for example, BERT's WordPiece, and Spacy's regular vocab),
            2. the token vectors for the sub word tokenization,
           Do min pooling over the subtokens.
        """
        super().__init__()
        self.sub_word_vocab = sub_word_vocab
        self.word_vocab = word_vocab
        self.sub_word_embedder = sub_word_embedder
        self.word_embedder = word_embedder
        self.sub_word_max_tokens_per_seq = sub_word_max_tokens_per_seq

    def forward(self, lstxt: T.List[str]) -> torch.Tensor:
        """
        Args:
            lstxt: A list of sentences.
        Returns:
            embedding: (B, L, E)
                L is computed like this:
                    Sentences are tokenized by self.sub_word_vocab.tokenizer, and truncated to
                    self.sub_word_max_tokens_per_seq.
                    The sentences are tokenized by self.word_vocab.tokenizer, and truncated to 
                    the last word whose complete sub word tokenization was not truncated
                    above.
                    L is the number of words in the sentence with the most word tokens.
        """
        self.subword_vocab.hi()
        raise NotImplementedError()

        """
        lslsword = self.word_vocab.batch_tokenize(lstxt)
        lslsword_id = self.word_vocab.batch_get_tok_ids(lslsword)
        lslslssub_word_id = [
            [self.sub_word_vocab.tokenize_and_get_tok_ids(word) for word in lsword]
            for lsword in lslsword
        ]

        # We might have a maximum length for the embedding allowed by the subword vocab
        # For example, if we're doing a BERT model, there's a maximum number of tokens.

        sub_word_ids = torch.tensor()
        subwords = self.sub_word_embedder.embed()
        """

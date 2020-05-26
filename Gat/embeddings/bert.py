import logging
from typing import List
from typing import Optional

import numpy as np
import torch
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoTokenizer

from .base import ContextualWordToVec

logging.basicConfig()
logger = logging.getLogger("embeddings")
logger.setLevel(logging.INFO)


class BertWordToVec(ContextualWordToVec):
    SPECIAL_TOKENS_AT_BEGIN: int = 1
    SPECIAL_TOKENS_AT_END: int = 1

    def __repr__(self) -> str:
        return "bert-base-uncased-last-four-layers"

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: int = -1,
        initial_sentences: Optional[List[List[str]]] = None,
        last_how_many_layers: int = 4,
    ):
        self.name = model_name
        self.model_name = model_name
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name)
        config.output_hidden_states = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name, config=config
        )
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name, config=config
        )
        self.last_how_many_layers = last_how_many_layers
        self.dim = 768 * self.last_how_many_layers

    def for_lsword(self, sent: List[str]) -> np.ndarray:
        """

        Returns
        -------
            np.ndarray: (len(sent), self.dim)
        """
        required_inputs = ["input_ids", "attention_mask", "token_type_ids"]
        inputs = self.tokenizer.encode_plus(
            " ".join(sent),
            add_special_tokens=True,
            return_tensors="pt",
            max_length=self.tokenizer.max_len,
        )

        # Figure out into how many sublsword each token was broken into
        spans = [len(self.tokenizer.tokenize(word)) for word in sent]
        logger.debug(f"Found spans: {spans}")
        assert (
            sum(spans) + self.SPECIAL_TOKENS_AT_END + self.SPECIAL_TOKENS_AT_BEGIN
        ) == len(inputs["input_ids"][0])

        inputs = {k: inputs[k] for k in required_inputs}
        with torch.no_grad():
            output = self.model(**inputs)
            all_hidden_states = output[2]

        # Each is of shape (1, len(tokens) + SPECIAL_TOKENS_AT_BEGIN + SPECIAL_TOKENS_AT_END, 768]
        last_layers = all_hidden_states[-self.last_how_many_layers :]
        concatenated_states = torch.cat(last_layers, dim=2).squeeze(dim=0).numpy()
        assert self.SPECIAL_TOKENS_AT_END > 0
        concatenated_states = concatenated_states[
            self.SPECIAL_TOKENS_AT_BEGIN : -self.SPECIAL_TOKENS_AT_END
        ]

        # Sum up the multiple tokens that belong to one word
        embeddings = np.zeros((len(sent), self.dim))

        word_end = 0
        for i, span in enumerate(spans):
            word_begin = word_end
            word_end = word_begin + span
            embed = concatenated_states[word_begin:word_end]
            embed = embed.mean(axis=0)
            assert len(embed) == self.dim
            embeddings[i, :] = embed

        return embeddings

    def for_lslsword(self, sents: List[List[str]]) -> np.ndarray:
        """

        Returns
        -------
            list(np.ndarray): Of length `len(sents)`
                Where each item is np.ndarray: (len(a_sent), self.dim)
        
        """
        return [self.for_lsword(sent) for sent in sents]

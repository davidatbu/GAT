from .configuration_utils import PretrainedConfig

class BertConfig(PretrainedConfig):
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    hidden_act: str
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    max_position_embeddings: int
    type_vocab_size: int
    initializer_range: float
    layer_norm_eps: float
    pad_token_id: int

__all__ = ["BertConfig"]

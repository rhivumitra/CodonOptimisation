from transformers import PretrainedConfig
import json
import torch


class StripedHyenaConfig(PretrainedConfig):
    model_type = "stripedhyena"

    def __init__(
        self,
        vocab_size=91,
        hidden_size=4096,
        num_filters=4096,
        inner_mlp_size=14336,
        attn_layer_idxs=[],
        hyena_layer_idxs=[],
        num_layers=32,
        tie_embeddings=False,
        short_filter_length=3,
        num_attention_heads=32,
        proj_groups=4,
        hyena_filter_groups=256,
        split_k0=True,
        column_split_hyena=True,
        column_split=False,
        model_parallel_size=1,
        pipe_parallel_size=1,
        short_filter_bias=True,
        mha_out_proj_bias=False,
        qkv_proj_bias=False,
        final_norm=True,
        use_cache=True,
        use_flash_attention_2=True,
        use_flash_rmsnorm=True,
        use_flash_depthwise=False,
        use_flashfft=False,
        inference_mode=False,
        prefill_style="fft",
        max_seqlen=32768,
        eps=1e-5,
        state_size=2,
        rotary_emb_base=500000,
        smeared_gqa=False,
        make_vocab_size_divisible_by=8,
        log_intermediate_values=False,
        compile=False,
        hyena_block_dtype=torch.float32,
        low_mem_mode=False,
        long_fir_threshold=None,
        depthwise_dtype=torch.bfloat16,
        mlp_dtype =torch.bfloat16,
        attn_block_dtype =torch.bfloat16,
        inner_size_multiple_of=64,
        padding_idx=0,
        process_group=None,
        max_batch_size=1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.inner_mlp_size = inner_mlp_size
        self.attn_layer_idxs = attn_layer_idxs
        self.hyena_layer_idxs = hyena_layer_idxs
        self.num_layers = num_layers
        self.tie_embeddings = tie_embeddings
        self.short_filter_length = short_filter_length
        self.num_attention_heads = num_attention_heads
        self.proj_groups = proj_groups
        self.hyena_filter_groups = hyena_filter_groups
        self.split_k0 = split_k0
        self.column_split_hyena = column_split_hyena
        self.column_split = column_split
        self.model_parallel_size = model_parallel_size
        self.pipe_parallel_size = pipe_parallel_size
        self.short_filter_bias = short_filter_bias
        self.mha_out_proj_bias = mha_out_proj_bias
        self.qkv_proj_bias = qkv_proj_bias
        self.final_norm = final_norm
        self.use_cache = use_cache
        self.use_flash_attention_2 = use_flash_attention_2
        self.use_flash_rmsnorm = use_flash_rmsnorm
        self.use_flash_depthwise = use_flash_depthwise
        self.use_flashfft = use_flashfft
        self.inference_mode = inference_mode
        self.prefill_style = prefill_style
        self.max_seqlen = max_seqlen
        self.eps = eps
        self.state_size = state_size
        self.rotary_emb_base = rotary_emb_base
        self.smeared_gqa = smeared_gqa
        self.make_vocab_size_divisible_by = make_vocab_size_divisible_by
        self.log_intermediate_values = log_intermediate_values
        self.compile = compile
        self.hyena_block_dtype = hyena_block_dtype
        self.low_mem_mode = low_mem_mode
        self.long_fir_threshold = long_fir_threshold
        self.depthwise_dtype = depthwise_dtype
        self.mlp_dtype = mlp_dtype
        self.attn_block_dtype = attn_block_dtype
        self.inner_size_multiple_of = inner_size_multiple_of
        self.padding_idx = padding_idx
        self.process_group = process_group
        self.max_batch_size = max_batch_size
        super().__init__(**kwargs)

    def to_dict(self):
        return {attr: getattr(self, attr) for attr in self.__dict__}

    @classmethod
    def from_original_config(cls, config_path, **kwargs):
        with open(config_path, "r") as f:
            config = json.load(f)

        return cls(**config, **kwargs)

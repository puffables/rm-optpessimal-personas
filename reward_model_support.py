import torch
from typing import Type
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import random
import numpy as np
import yaml

class RewardModel:
    MODEL_REGISTRY = {}

    _DTYPE_MAP = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }

    """Base class for all model subclasses with shared initialization logic."""
    @classmethod
    def create(cls, model_name, **kwargs):
        model_class = cls.MODEL_REGISTRY.get(model_name)
        if model_class is not None:
            return model_class(model_name, **kwargs)
        return cls(model_name, **kwargs)

    @staticmethod
    def _load_model_config(model_name):
        """Load model configuration from config/reward_models.yaml."""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'reward_models.yaml')
        with open(config_path, 'r') as f:
            models = yaml.safe_load(f)
        for entry in models:
            if entry['name'] == model_name:
                return entry
        raise ValueError(f"Model '{model_name}' not found in {config_path}")

    @staticmethod
    def _parse_model_params(params):
        """Convert YAML params dict to kwargs suitable for from_pretrained()."""
        if not params:
            return {}
        result = {}
        for k, v in params.items():
            if k == 'torch_dtype':
                result[k] = RewardModel._DTYPE_MAP[v]
            else:
                result[k] = v
        return result

    def __init__(self, model_name: str, **kwargs):
        config = self._load_model_config(model_name)
        self.multi_gpu = config['multi_gpu']
        self.default_batch_size = config['batch_size']
        self.model_params = self._parse_model_params(config.get('params'))

        self.name = model_name
        self._initialize_url()
        self._initialize_model()
        self._set_deterministic_mode()

        if self.multi_gpu:
            self._setup_multi_gpu()

    def _extract_scores_from_outputs(self, outputs):
        """Extract scores from model outputs. Individual models must override this."""
        raise NotImplementedError("Subclasses must implement _extract_scores_from_outputs")

    def _initialize_url(self):
        self.url = f"""https://huggingface.co/{self.name}"""

    def _initialize_model(self):
        """Initialize the model and tokenizer with appropriate settings."""
        model_params = getattr(self, 'model_params', {})

        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print(f"Found {n_gpus} CUDA devices")
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            print("Using MPS device")
            self.device = 'mps'
        else:
            print("Using CPU")
            self.device = 'cpu'

        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.tokenizer_class = type(self.tokenizer).__name__

        if self.multi_gpu:
            self.device_map = 'auto'
            print(f"Using batch size {self.default_batch_size} with multi-GPU setup")
        else:
            self.device_map = self.device
            print(f"Using batch size {self.default_batch_size} with single GPU setup ({self.device})")

        if self.device == 'cuda' and torch.cuda.device_count() > 1:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.name,
                device_map=self.device_map,
                **model_params
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.name,
                **model_params
            ).to(self.device)

        print(f"Model dtype: {next(self.model.parameters()).dtype}")
        print(f"Reward model {self.name} (tokenizer class {self.tokenizer_class}) initialized")
        if self.device == 'cuda' and hasattr(self.model, 'hf_device_map'):
            print(f"Model devices: {self.model.hf_device_map}")

    def _setup_multi_gpu(self):
        """Handle multi-GPU specific setup after model initialization."""
        if not hasattr(self.model, 'hf_device_map'):
            print("Warning: Model does not have device mapping information")
            return

        print("\nModel device mapping:")
        for name, device in self.model.hf_device_map.items():
            print(f"{name}: {device}")

        final_device = None
        for layer_name in ['regression_layer', 'classifier', 'score_head', 'lm_head']:
            if layer_name in self.model.hf_device_map:
                final_device = self.model.hf_device_map[layer_name]
                break

        if final_device is not None:
            print(f"\nFinal layers are on cuda:{final_device}")

            params_to_move = []
            for name, param in self.model.named_parameters():
                if any(key in name for key in ['transform_matrix', 'output_layer', 'final_layer']):
                    target_device = f'cuda:{final_device}'
                    if str(param.device) != target_device:
                        params_to_move.append((name, param, target_device))

            for name, param, target_device in params_to_move:
                with torch.no_grad():
                    new_param = torch.nn.Parameter(
                        param.data.to(target_device),
                        requires_grad=param.requires_grad
                    )
                    path = name.split('.')
                    module = self.model
                    for part in path[:-1]:
                        module = getattr(module, part)
                    delattr(module, path[-1])
                    module.register_parameter(path[-1], new_param)
                    print(f"Moved {name} to {target_device}")

    def _set_deterministic_mode(self, seed=42):
        """Set deterministic mode for reproducibility."""
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        self.model.eval()

    def _calculate_batch_scores(self, formatted_convs):
        """Calculate reward scores for a batch of formatted conversations."""
        tokenized_inputs = self.tokenizer(
            formatted_convs,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        if self.multi_gpu:
            embed_device = 'cuda:0'
            tokenized_inputs = {k: v.to(embed_device) for k, v in tokenized_inputs.items()}
        else:
            tokenized_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}

        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)
            batch_scores = self._extract_scores_from_outputs(outputs)

        return batch_scores

    def get_reward_scores_from_response_token_ids(self, prompt, response_token_ids, max_gpu_batch):
        """Calculate reward scores for multiple response tokens."""
        all_scores = []

        for i in range(0, len(response_token_ids), max_gpu_batch):
            batch_token_ids = response_token_ids[i:i + max_gpu_batch]

            conversations = [
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": self.tokenizer.decode([token_id])}
                ]
                for token_id in batch_token_ids
            ]

            formatted_convs = [
                self.tokenizer.apply_chat_template(conv, tokenize=False)
                for conv in conversations
            ]

            batch_scores = self._calculate_batch_scores(formatted_convs)
            all_scores.extend(batch_scores)

        return all_scores

    def get_reward_scores_from_response_token_ids_fixed(self, prompt, response_token_ids, max_gpu_batch):
        """Same behavior as get_reward_scores_from_response_token_ids, with
        the two tokenization bugs fixed but WITHOUT KV caching (full
        sequence re-run per candidate, same as the original) — isolates the
        effect of the tokenization fix alone, independent of caching.

        Fixes:
          1. add_special_tokens=False, so the already-chat-templated string
             (which already contains a literal BOS) doesn't get a second BOS
             prepended by the tokenizer.
          2. Response tokens are appended as raw token IDs instead of being
             decoded to text and the whole conversation re-tokenized, which
             can silently substitute a different token for ~half the
             vocabulary (BPE merges at the seam between the decoded text and
             surrounding template text).
        """
        prefix_ids, suffix_ids = self._build_prefix_suffix_ids(prompt)
        prefix_ids = prefix_ids.to(self.device)
        suffix_ids = suffix_ids.to(self.device)

        all_scores = []
        for i in range(0, len(response_token_ids), max_gpu_batch):
            batch_ids = response_token_ids[i:i + max_gpu_batch]
            batch_size = len(batch_ids)

            token_col = torch.tensor(batch_ids, device=self.device).unsqueeze(1)
            full_ids = torch.cat([
                prefix_ids.expand(batch_size, -1),
                token_col,
                suffix_ids.expand(batch_size, -1),
            ], dim=1)

            with torch.no_grad():
                outputs = self.model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids))
                batch_scores = self._extract_scores_from_outputs(outputs)
            all_scores.extend(batch_scores)

        return all_scores

    def _build_prefix_suffix_ids(self, prompt):
        """Split the chat-templated conversation into the tokens before and
        after the assistant response, by templating with a sentinel string
        and splitting on it at the text level (not by concatenating
        separately-tokenized fragments, which risks BPE merges at the seam).
        """
        sentinel = "RM_KV_CACHE_SENTINEL"
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": sentinel},
        ]
        full_text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        prefix_text, suffix_text = full_text.split(sentinel)

        prefix_ids = self.tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt").input_ids
        suffix_ids = self.tokenizer(suffix_text, add_special_tokens=False, return_tensors="pt").input_ids
        return prefix_ids, suffix_ids

    def get_reward_scores_from_response_token_ids_kv_cached(self, prompt, response_token_ids, max_gpu_batch,
                                                             reproduce_bos_bug=False):
        """KV-cache variant of get_reward_scores_from_response_token_ids.

        Encodes the (identical) prompt prefix once, caches its key/value
        states, and for each candidate response token only runs that one new
        token (plus the fixed template suffix) through the model, reusing the
        cached prefix attention instead of recomputing it per candidate.
        Only supports the single-device path (multi_gpu=False models).

        reproduce_bos_bug: if True, prepends an extra BOS token to the prefix,
        matching the double-BOS artifact in get_reward_scores_from_response_token_ids
        (which tokenizes the already-chat-templated string with
        add_special_tokens=True). Lets you isolate the caching speedup from
        that specific tokenization bug. Note there is no equivalent toggle for
        the decode()-then-re-tokenize round-trip corruption in the baseline
        method — that bug is structurally incompatible with caching (it
        depends on re-tokenizing variable-length text per candidate, which
        caching replaces by construction with appending known token IDs).
        """
        from transformers.cache_utils import DynamicCache

        prefix_ids, suffix_ids = self._build_prefix_suffix_ids(prompt)
        prefix_ids = prefix_ids.to(self.device)
        suffix_ids = suffix_ids.to(self.device)

        if reproduce_bos_bug:
            extra_bos = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)
            prefix_ids = torch.cat([extra_bos, prefix_ids], dim=1)

        prefix_len = prefix_ids.shape[1]

        with torch.no_grad():
            prefix_out = self.model(
                input_ids=prefix_ids,
                attention_mask=torch.ones_like(prefix_ids),
                past_key_values=DynamicCache(),
                use_cache=True,
            )
        prefix_cache = prefix_out.past_key_values

        all_scores = []
        for i in range(0, len(response_token_ids), max_gpu_batch):
            batch_ids = response_token_ids[i:i + max_gpu_batch]
            batch_size = len(batch_ids)

            token_col = torch.tensor(batch_ids, device=self.device).unsqueeze(1)
            new_ids = torch.cat([token_col, suffix_ids.expand(batch_size, -1)], dim=1)
            new_len = new_ids.shape[1]

            expanded_cache = DynamicCache()
            expanded_cache.key_cache = [k.repeat_interleave(batch_size, dim=0) for k in prefix_cache.key_cache]
            expanded_cache.value_cache = [v.repeat_interleave(batch_size, dim=0) for v in prefix_cache.value_cache]
            expanded_cache._seen_tokens = prefix_cache._seen_tokens

            attention_mask = torch.ones(batch_size, prefix_len + new_len, device=self.device)
            position_ids = torch.arange(prefix_len, prefix_len + new_len, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=new_ids,
                    attention_mask=attention_mask,
                    past_key_values=expanded_cache,
                    position_ids=position_ids,
                    use_cache=False,
                )
                batch_scores = self._extract_scores_from_outputs(outputs)
            all_scores.extend(batch_scores)

        return all_scores

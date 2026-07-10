"""
Generate base-model next-token logits for persona-conditioned prompts.

For each active model in config/llama_base_models.yaml, and for each (template, persona)
combination defined in config/persona_prompts.yaml and config/personas.yaml, runs a single
forward pass to get the full-vocabulary logit distribution over the next token, and saves
results to data/persona_base_model_logits/<model>.csv (one column per template/persona
combination, same token_id/token_name/token_decoded format as data/base_model_logits/).

Unlike the reward-model sweep (generate_persona_reward_model_scores.py), this doesn't need
to batch over every candidate token — a causal LM's logits already score every possible next
token in one forward pass over the prompt.

Checkpoints after every (template, persona) combination, so an interrupted run can be
resumed by simply re-running the script — already-scored columns are skipped.
"""

import argparse
import re
import yaml
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

SCRIPT_ROOT = Path(__file__).parent
CONFIG_DIR = SCRIPT_ROOT / 'config'
OUTPUT_DIR = SCRIPT_ROOT / 'data' / 'persona_base_model_logits'

DTYPE_MAP = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--templates', type=str, default=None,
                     help='Comma-separated template names to run (default: all in persona_prompts.yaml)')
parser.add_argument('--models', type=str, default=None,
                     help='Comma-separated model names to run (default: all in llama_base_models.yaml)')
parser.add_argument('--personas', type=str, default=None,
                     help='Comma-separated persona slugs to run (default: all in personas.yaml)')
parser.add_argument('--max-personas', type=int, default=None,
                     help='Only run the first N personas (for quick pilots)')
args = parser.parse_args()


def slugify(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


with open(CONFIG_DIR / 'llama_base_models.yaml') as f:
    models = yaml.safe_load(f)

if args.models:
    selected = set(args.models.split(','))
    models = [m for m in models if m['name'] in selected]

with open(CONFIG_DIR / 'persona_prompts.yaml') as f:
    prompt_templates = yaml.safe_load(f)['templates']

if args.templates:
    selected = args.templates.split(',')
    prompt_templates = {k: v for k, v in prompt_templates.items() if k in selected}

with open(CONFIG_DIR / 'personas.yaml') as f:
    personas = yaml.safe_load(f)['personas']

if args.personas:
    selected = set(args.personas.split(','))
    personas = [p for p in personas if slugify(p['name']) in selected]

if args.max_personas:
    personas = personas[:args.max_personas]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def template_for_pretrained_model(tok, prompt):
    prompt_string = f"User: {prompt}\nAssistant:"
    return tok(prompt_string, return_tensors='pt')


def template_for_instruction_tuned_model(tok, prompt):
    messages = [{'role': 'user', 'content': prompt}]
    return tok.apply_chat_template(
        messages, return_tensors='pt', return_dict=True, add_generation_prompt=True)


for model_info in models:
    model_name = model_info['name']
    safe_name = model_name.replace('/', '--')
    output_path = OUTPUT_DIR / f"{safe_name}.csv"

    df = None
    existing_columns = set()
    if output_path.exists():
        df = pd.read_csv(output_path)
        existing_columns = set(df.columns)

    pending = []
    for template_name in prompt_templates:
        for persona in personas:
            col = f"{template_name}__{slugify(persona['name'])}"
            if col not in existing_columns:
                pending.append((template_name, persona, col))

    if not pending:
        print(f"Skipping {model_name} — all persona/template combinations already scored")
        continue

    print(f"Processing model: {model_name} ({len(pending)} persona/template combinations to score)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=DTYPE_MAP[model_info['dtype']],
        device_map='auto'
    ).eval()
    device = model.get_input_embeddings().weight.device

    vocab = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
    token_names, token_ids = zip(*vocab)
    token_decoded = tokenizer.batch_decode([[tid] for tid in token_ids], skip_special_tokens=False)

    if df is None:
        df = pd.DataFrame({
            'token_id': token_ids,
            'token_name': token_names,
            'token_decoded': token_decoded,
        })

    for template_name, persona, col in tqdm(pending, desc=model_name):
        prompt_text = prompt_templates[template_name]['text'].format(persona=persona['name'])

        if model_info['type'] == 'pretrained':
            model_inputs = template_for_pretrained_model(tokenizer, prompt_text)
        elif model_info['type'] == 'instruction-tuned':
            model_inputs = template_for_instruction_tuned_model(tokenizer, prompt_text)
        else:
            raise ValueError(f"Unknown model type: {model_info['type']}")

        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        with torch.inference_mode():
            outputs = model(**model_inputs, use_cache=False)
            last_token_logits = outputs.logits[0, -1, :].float()

        logits_list = last_token_logits.cpu().numpy().tolist()
        df[col] = [logits_list[tid] for tid in token_ids]
        df.to_csv(output_path, index=False)  # checkpoint after every persona/template

    print(f"Saved {safe_name}.csv")

    del model
    del tokenizer
    torch.cuda.empty_cache()

print("Done.")

#!/usr/bin/env python3
"""Phase 1: Extract ICR scores from LLM using single forward pass.

Uses Qwen2.5-7B-Instruct with HaluEval data. Since HaluEval provides
pre-existing answers + labels, we use a single forward pass (not generation)
to extract hidden states and attentions. Due to causal attention mask,
hidden states at each answer token position are identical to autoregressive
generation, making this ~10x faster.

Output:
  - icr_score.pt: dict {sample_id: list[L][N_answer_tokens]} of ICR scores
  - output_judge.jsonl: one line per sample {"id": int, "result_type": 0|1}
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

from src.icr_score import ICRScore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ICR scores from LLM")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_dir", type=str, default="data/halueval")
    parser.add_argument("--save_dir", type=str, default="saves/halueval")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    return parser.parse_args()


def load_data(data_dir: str, split: str):
    """Load HaluEval JSONL data."""
    samples = []
    files = []
    if split in ("train", "all"):
        files.append(os.path.join(data_dir, "train_qa_samples.json"))
    if split in ("test", "all"):
        files.append(os.path.join(data_dir, "test_qa_samples.json"))

    for fpath in files:
        logger.info(f"Loading {fpath}")
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def format_messages(sample):
    """Format a HaluEval sample into Qwen2.5 chat messages."""
    system_msg = {"role": "system", "content": "You are a helpful assistant."}
    user_msg = {
        "role": "user",
        "content": f"Knowledge: {sample['knowledge']}\nQuestion: {sample['question']}",
    }
    assistant_msg = {"role": "assistant", "content": sample["answer"]}
    return system_msg, user_msg, assistant_msg


def compute_core_positions(tokenizer, system_msg, user_msg, assistant_msg):
    """Compute core_positions dict for ICRScore.

    Returns:
        core_positions: dict with user_prompt_start, user_prompt_end, response_start
        prompt_ids: token ids for prompt only (with generation prompt)
        full_ids: token ids for full sequence including answer
    """
    # Prompt without answer (add_generation_prompt=True adds "<|im_start|>assistant\n")
    messages_no_answer = [system_msg, user_msg]
    prompt_ids = tokenizer.apply_chat_template(
        messages_no_answer, tokenize=True, add_generation_prompt=True
    )

    # Full sequence with answer
    messages_full = [system_msg, user_msg, assistant_msg]
    full_ids = tokenizer.apply_chat_template(messages_full, tokenize=True)

    response_start = len(prompt_ids)

    # Find user section boundaries using special tokens
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    im_start_positions = [i for i, t in enumerate(full_ids) if t == im_start_id]
    im_end_positions = [i for i, t in enumerate(full_ids) if t == im_end_id]

    # im_start_positions[0] = system, [1] = user, [2] = assistant
    user_prompt_start = im_start_positions[1]
    user_prompt_end = im_end_positions[1] + 1  # after <|im_end|> of user

    core_positions = {
        "user_prompt_start": user_prompt_start,
        "user_prompt_end": user_prompt_end,
        "response_start": response_start,
    }

    return core_positions, prompt_ids, full_ids


def restructure_hidden_states(model_hidden_states, input_len):
    """Restructure model hidden states into ICRScore format.

    Model outputs hidden_states as tuple of (L+1) tensors,
    each [1, seq_len, hidden_size].

    ICRScore expects:
      hidden_states[0] = [hs_layer_l[:, :input_len, :] for l in 0..L]
      hidden_states[t] = [hs_layer_l[:, input_len+t-1:input_len+t, :] for l in 0..L]
      for t = 1..answer_len
    """
    num_layers = len(model_hidden_states)
    seq_len = model_hidden_states[0].shape[1]
    answer_len = seq_len - input_len

    # hidden_states[0]: input part — list of layer tensors [1, input_len, hidden_size]
    hs_list = []
    hs_input = [model_hidden_states[l][:, :input_len, :] for l in range(num_layers)]
    hs_list.append(hs_input)

    # hidden_states[t] for t=1..answer_len: each answer token
    for t in range(answer_len):
        pos = input_len + t
        hs_token = [model_hidden_states[l][:, pos : pos + 1, :] for l in range(num_layers)]
        hs_list.append(hs_token)

    return hs_list


def restructure_attentions(model_attentions, input_len):
    """Restructure model attentions into ICRScore format.

    Model outputs attentions as tuple of L tensors,
    each [1, num_heads, seq_len, seq_len].

    ICRScore expects:
      attentions[0] = [attn_layer_l[:, :, :input_len, :input_len] for l in 0..L-1]
      attentions[t] = [attn_layer_l[:, :, input_len+t-1:input_len+t, :input_len+t] for l]
      for t = 1..answer_len
    """
    num_layers = len(model_attentions)
    seq_len = model_attentions[0].shape[2]
    answer_len = seq_len - input_len

    attn_list = []
    # attentions[0]: input part
    attn_input = [
        model_attentions[l][:, :, :input_len, :input_len] for l in range(num_layers)
    ]
    attn_list.append(attn_input)

    # attentions[t] for t=1..answer_len
    for t in range(answer_len):
        pos = input_len + t
        attn_token = [
            model_attentions[l][:, :, pos : pos + 1, : pos + 1] for l in range(num_layers)
        ]
        attn_list.append(attn_token)

    return attn_list


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded")

    # Load data
    samples = load_data(args.data_dir, args.split)

    # Check for existing checkpoint
    ckpt_path = save_dir / "checkpoint.pt"
    if ckpt_path.exists():
        logger.info("Loading checkpoint...")
        checkpoint = torch.load(ckpt_path, weights_only=False)
        icr_scores_dict = checkpoint["icr_scores"]
        labels_dict = checkpoint["labels"]
        start_idx = checkpoint["next_idx"]
        logger.info(f"Resuming from sample {start_idx}")
    else:
        icr_scores_dict = {}
        labels_dict = {}
        start_idx = 0

    icr_device = "cuda:0"
    batch_size = args.batch_size
    total_samples = len(samples)
    start_time = time.time()
    processed_count = len(icr_scores_dict)

    # Ensure tokenizer has pad token for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for batch_start in range(start_idx, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)

        # --- Prepare batch: tokenize, filter, collect metadata ---
        batch_input_ids = []
        batch_core_positions = []
        batch_seq_lens = []
        batch_indices = []

        for idx in range(batch_start, batch_end):
            if idx in icr_scores_dict:
                continue

            sample = samples[idx]
            system_msg, user_msg, assistant_msg = format_messages(sample)
            core_positions, prompt_ids, full_ids = compute_core_positions(
                tokenizer, system_msg, user_msg, assistant_msg
            )

            # Truncate if too long
            if len(full_ids) > args.max_seq_len:
                full_ids = full_ids[: args.max_seq_len]
                if core_positions["response_start"] >= args.max_seq_len:
                    logger.warning(f"Sample {idx}: prompt alone exceeds max_seq_len, skipping")
                    continue

            input_len = core_positions["response_start"]
            answer_len = len(full_ids) - input_len

            if answer_len <= 0:
                logger.warning(f"Sample {idx}: no answer tokens after truncation, skipping")
                continue

            batch_input_ids.append(full_ids)
            batch_core_positions.append(core_positions)
            batch_seq_lens.append(len(full_ids))
            batch_indices.append(idx)

        if not batch_input_ids:
            continue

        # --- Pad sequences (right padding) and build attention mask ---
        max_len = max(batch_seq_lens)
        pad_id = tokenizer.pad_token_id

        padded_ids = []
        attention_masks = []
        for ids in batch_input_ids:
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)

        input_ids_tensor = torch.tensor(padded_ids, dtype=torch.long).to(model.device)
        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long).to(model.device)

        # --- Batched forward pass ---
        with torch.no_grad():
            outputs = model(
                input_ids_tensor,
                attention_mask=attention_mask_tensor,
                output_hidden_states=True,
                output_attentions=True,
            )

        # --- Per-sample ICR computation ---
        for b, idx in enumerate(batch_indices):
            seq_len = batch_seq_lens[b]
            core_pos = batch_core_positions[b]
            input_len = core_pos["response_start"]

            # Extract per-sample hidden states & attentions (strip padding)
            sample_hs = tuple(
                hs[b : b + 1, :seq_len, :] for hs in outputs.hidden_states
            )
            sample_attn = tuple(
                attn[b : b + 1, :, :seq_len, :seq_len] for attn in outputs.attentions
            )

            hidden_states = restructure_hidden_states(sample_hs, input_len)
            attentions = restructure_attentions(sample_attn, input_len)

            icr = ICRScore(
                hidden_states=hidden_states,
                attentions=attentions,
                core_positions=core_pos,
                icr_device=icr_device,
            )
            icr_scores_item, _ = icr.compute_icr(
                top_k=20,
                top_p=0.1,
                pooling="mean",
                attention_uniform=False,
                hidden_uniform=False,
                use_induction_head=True,
            )

            icr_scores_dict[idx] = icr_scores_item
            label = 1 if samples[idx]["hallucination"] == "no" else 0
            labels_dict[idx] = label

            del sample_hs, sample_attn, hidden_states, attentions, icr

        # --- Clean up batch ---
        del outputs, input_ids_tensor, attention_mask_tensor
        torch.cuda.empty_cache()

        # --- Log progress ---
        processed_count += len(batch_indices)
        elapsed = time.time() - start_time
        rate = elapsed / max(processed_count, 1)
        remaining = rate * (total_samples - batch_end)
        logger.info(
            f"[{batch_end}/{total_samples}] "
            f"batch={len(batch_indices)}, "
            f"{rate:.2f}s/sample, "
            f"ETA: {remaining / 60:.1f}min"
        )

        # --- Checkpoint ---
        if batch_end % args.checkpoint_every < batch_size or batch_end >= total_samples:
            logger.info(f"Saving checkpoint at sample {batch_end}...")
            torch.save(
                {
                    "icr_scores": icr_scores_dict,
                    "labels": labels_dict,
                    "next_idx": batch_end,
                },
                ckpt_path,
            )

    # Save final results
    logger.info("Saving final results...")
    torch.save(icr_scores_dict, save_dir / "icr_score.pt")

    with open(save_dir / "output_judge.jsonl", "w") as f:
        for sample_id in sorted(labels_dict.keys()):
            f.write(
                json.dumps({"id": sample_id, "result_type": labels_dict[sample_id]}) + "\n"
            )

    # Clean up checkpoint
    if ckpt_path.exists():
        ckpt_path.unlink()

    elapsed_total = time.time() - start_time
    logger.info(
        f"Done! Processed {len(icr_scores_dict)} samples in {elapsed_total / 60:.1f} min"
    )


if __name__ == "__main__":
    main()

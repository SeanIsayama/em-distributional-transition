from __future__ import annotations

import torch


def get_layer(model, layer_idx: int):
    """Return the transformer layer module, unwrapping PEFT if needed.

    Path: model.base_model.model (Qwen2ForCausalLM) → .model.layers[i],
    matching the direct access pattern used in the original notebooks.
    """
    base = model.base_model.model if hasattr(model, "base_model") else model
    return base.model.layers[layer_idx]


def extract_response_activations(
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int,
    n_responses: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[torch.Tensor, list[list[str]]]:
    """Generate responses, extract activations, and decode text in one pass.

    For each prompt, generates n_responses independently, then runs a single
    forward pass on each complete (prompt + response) token sequence and
    mean-pools the hidden states at the response-token positions.  Decoding
    is done from the same ``out_ids`` used for the activation pass, so the
    returned texts and activations correspond to identical generations.

    Prompt length is computed per-prompt from the chat template so that
    prompts of different lengths are sliced correctly.

    Returns
    -------
    activations : torch.Tensor
        Shape (n_prompts, n_responses, hidden_dim) in bfloat16.
    texts : list[list[str]]
        Decoded response strings; ``texts[i][j]`` matches ``activations[i, j]``.
    """
    layer = get_layer(model, layer_idx)
    device = next(model.parameters()).device
    act_results: list[torch.Tensor] = []
    text_results: list[list[str]] = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        prompt_acts: list[torch.Tensor] = []
        prompt_texts: list[str] = []

        for _ in range(n_responses):
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            # out_ids: (1, prompt_len + response_len) — single sequence, no padding

            text = tokenizer.decode(out_ids[0, prompt_len:], skip_special_tokens=True)
            prompt_texts.append(text)

            captured: list[torch.Tensor] = []

            def _hook(module, inp, out, _buf=captured):
                h = out[0] if isinstance(out, tuple) else out
                _buf.append(h.detach())

            handle = layer.register_forward_hook(_hook)
            with torch.no_grad():
                model(out_ids)
            handle.remove()

            h = captured[-1]  # (1, full_len, hidden_dim)
            response_h = h[0, prompt_len:, :]  # (response_len, hidden_dim)
            if response_h.shape[0] == 0:
                act = torch.zeros(h.shape[-1], dtype=torch.bfloat16, device=h.device)
            else:
                act = response_h.mean(dim=0).to(torch.bfloat16)
            prompt_acts.append(act)

        act_results.append(torch.stack(prompt_acts))  # (n_responses, hidden_dim)
        text_results.append(prompt_texts)

    return torch.stack(act_results), text_results  # (n_prompts, n_responses, hidden_dim)

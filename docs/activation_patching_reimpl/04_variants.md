# Activation Patching Variants

Same intervention primitive; different experimental wrappers. Implement the core loop once, then swap how you build sources/targets and which module you write.

---

## 1. Clean → Corrupt → Restore (causal tracing)

**Use when:** localizing which `(token, layer)` mediates a factual prediction.

### Procedure

```text
1. Clean:  forward(x)           → cache ALL residuals h_clean[L, t]
2. Corrupt: forward(x*)         → measure P_corrupt(y)
3. For each (t, L):
     forward(x*) with h[L, t] ← h_clean[L, t]
     IE[t, L] = P_patched(y) - P_corrupt(y)
```

### Corruption options

| Method | How | Notes |
|--------|-----|-------|
| Gaussian noise | Add \(\mathcal{N}(0, (3\sigma)^2)\) to subject embeddings | Classic Meng et al. |
| Counterfactual subject | Replace subject string | Cleaner semantics |
| Wrong-fact prompt | Change attribute context | Task-dependent |

### Window restore for MLP / Attn

Single MLP outputs are small residual updates. Restore layers `[L - w, L + w]` of the **module output** (not necessarily full residual):

```python
for L in range(center - w, center + w + 1):
    mlp_out[L][:, t] = clean_mlp[L][:, t]
```

### Metric

\[
\text{AIE}(t,L) = \mathbb{E}\big[P(y\mid \text{restore }t,L) - P(y\mid \text{corrupt})\big]
\]

Heatmap over tokens × layers is the deliverable.

---

## 2. Patchscopes (latent → readout prompt)

**Use when:** you want a linguistic readout of a hidden state.

### Procedure (matches `patchscope_lens`)

```python
# 1) Cache last-token states of source prompts: latents[L][b]
latents = collect_activations(nn_model, source_prompts)  # [L, B, d]

# 2) Fixed readout prompt, e.g. repeat_prompt():
#    "king king\n1135 1135\nhello hello\n?"
#    index_to_patch = -1  (# the '?')

# 3) For each layer L: run readout batch, write latents[L] into '?'
for L in range(n_layers):
    with nn_model.trace(readout_prompts):
        get_layer_output(nn_model, L)[arange(B), index_to_patch] = latents[L]
        probs[L] = get_next_token_probs(nn_model)
```

### Design knobs

- **Same-layer write** (above) vs always write into a fixed target layer.
- Custom readout (“The meaning of this is:”) vs repeat-prompt.
- `patchscope_generate` for multi-token string completions after the patch.

---

## 3. Attention-output patching

**Use when:** testing whether the **attention branch** (not full residual) carries the concept.

```python
# collect
hiddens = collect_activations(
    nn_model, source_prompts, get_activations=get_attention_output
)
# patch window of attention outputs
for patch_idx in range(layer, min(layer + k, n_layers)):
    get_attention_output(nn_model, patch_idx)[b, index] = hiddens[patch_idx]
```

See `patch_attention_lens` in `src/interventions.py`.

---

## 4. Object-attention input patching

**Use when:** forcing attention at later layers to **see** the source object state as if it were the target object token.

```python
# Source: capture hidden_states fed into attention at each layer (last token)
def get_act(model, layer):
    return get_attention(model, layer).input[1]["hidden_states"]

source_hiddens = collect_activations(..., get_activations=get_act)

# Target: overwrite attn input at object index for a window of layers
get_attention(nn_model, L).input[1]["hidden_states"][:, attn_idx_patch] = source_hiddens[L]
```

See `patch_object_attn_lens`. Framework-specific: nnsight’s `.input` API; in raw PyTorch, hook the attention module’s forward and replace the `hidden_states` argument.

---

## 5. Latent prompts (multi-slot patching)

**Use when:** composing several latent vectors into one forward (multi-concept prompts).

```python
# LatentPrompt has several placeholder positions (bos or special token)
# At each spot, write the corresponding source vector for layers
# [patch_from_layer, patch_until_layer]
for spot in latent_spots:
    for layer in range(patch_from_layer, patch_until_layer + 1):
        get_layer_output(nn_model, layer)[i, spot] = latents[layer][h_index]
```

See `run_latent_prompt` / `latent_prompt_lens`.

---

## 6. Steering (additive, not replacement)

Not strict patching, but same hooks:

```python
get_layer_output(nn_model, layer)[:, position] += factor * steering_vector
```

Use for continuous control; pair with mean-activation baselines (`get_mean_activations` in `exp_tools.py`).

---

## 7. Layer skip

```python
get_layer_output(nn_model, layer)[:, position] = get_layer_input(nn_model, layer)[:, position]
```

Ablates the layer’s contribution at that position (identity residual). Useful control: “is the layer doing anything here?”

---

## 8. Choosing a variant quickly

```text
Localize a fact in one prompt?     → Clean/Corrupt/Restore heatmap
Transfer a concept across prompts? → Object residual patching (doc 03)
Decode what a vector "means"?      → Patchscopes
Is it specifically attention?      → Attn-out or attn-input patch
Compose multiple latents?          → Latent prompts
Soft control?                      → Steering
```

Next: [05_indexing_metrics_pitfalls.md](05_indexing_metrics_pitfalls.md)

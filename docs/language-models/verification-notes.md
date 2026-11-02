# FLoRIST verification notes

These notes record what the official FLoRIST code actually does, compared with the paper, for the language-models module.

- **Paper:** Ramesh & Dass, *FLoRIST: Singular Value Thresholding for Efficient and Accurate Federated Fine-Tuning of Large Language Models*, MLSys 2026 (camera-ready).
- **Code:** https://github.com/DASS-Lab-Group/FLoRIST, commit `100d373025866d148e092a17001e1232dabeeabd` (2026-10-13), dependencies `peft==0.13.2`, `transformers==4.57.1`, `torch==2.4.1`.
- **Method:** the code was read, not run, except where a "replica" is mentioned (a NumPy copy of the relevant lines). File references are to that commit.
- **Status labels:** **Answered** (backed by code or data), **Partly answered**, **Unresolved**.

The app follows the paper's **Algorithm 1**. Where the paper and code differ, the app shows the algorithm and notes the difference here.

---

## 1. Zero-padding (rank matching) — Answered (by reading the code)

When `p < r_k`, Algorithm 1 pads `B_g` with zero columns and `A_g` with zero rows. When both factors are zero padded, the added components remain inactive during that round's local training under ordinary gradient-based optimization. The padded columns of `B` and rows of `A` receive zero gradient (`∂L/∂B = G Aᵀ` and `∂L/∂A = Bᵀ G`). A client with `r_k > p` therefore trains an adapter of rank at most `p` in that round, leaving its additional capacity unused. However, independently trained client updates can aggregate into a higher-rank update, so the global rank selected in a later round may increase.

- In a round that starts from a global adapter of rank `p`, each client's adapter has rank at most `min(p, r_k)`. Round 1 is the exception (`B_k = 0`, random `A_k`): clients use their full rank `r_k`.
- The stacked update then has rank at most `Σ_k min(p, r_k)`, which can exceed `p`. The next selected rank can increase, stay the same or decrease.
- This does not hold if anything makes the padded entries nonzero: random padding initialization, noise injection, or some LoRA variants.

**Official code path:**
- `main.py:268-290` `homogenize_lora_weights` zero-pads both factors (`p < r_k`) or keeps the first `r_k` components (`p > r_k`), as in Algorithm 1.
- `main.py:432-463`: from round 2 on, each client loads the padded adapters with `PeftModel.from_pretrained(..., is_trainable=True)`.
- `fed_utils/client.py:45-72`: a new `transformers.Trainer` is built for every client in every round, with `optim="adamw_torch"`, no `weight_decay` argument (default 0.0), `warmup_steps=0` and `fp16=True`. So AdamW starts with zero moment estimates each round. A zero gradient gives a zero update, and weight decay is off.
- LoRA dropout (0.05) acts on the layer input, so it doesn't make padded entries nonzero.
- **Conclusion:** in the official runs, padded components should stay exactly zero within a round. This comes from reading the code; it hasn't been confirmed by running it.

## 2. Initialization of `A` — Partly answered

- **Algorithm 1:** round 1 uses `B_k = 0` and `A_k ∼ N(0, σ²)`.
- **Code:** FLoRIST round 1 (`main.py:412-431`) creates each client's adapters with `get_peft_model(model, config)` and no `init_lora_weights` setting. That is PEFT's default initialization: `B = 0` and `A` drawn from a **Kaiming-uniform** distribution (PEFT 0.13.2), **not a normal distribution**.
- **Shared seed:** there's no explicit per-client seed for the LoRA initialization. The only `torch.manual_seed` call is in the FFA-LoRA path (`main.py:582`). Client selection reseeds NumPy with the round number (`fed_utils/client_participation_scheduling.py:5`).
  - `transformers.Trainer` normally calls `set_seed(args.seed)` (default 42) when it's constructed. That happens after each client's adapters are created.
  - So client 1's `A` depends on the RNG state left over from model loading, and later clients' `A` depend on the reseed plus whatever random numbers the previous client's training used.
  - In practice `A` differs across clients, but not by design. **Unresolved:** confirm by running.
- **App setting:** the toy experiment documents its own choice (shared or per-client seed) as a setting, as README_FL requires.

## 3. Data partitioning — Partly answered

The paper says all datasets use Dirichlet(α = 0.5) non-IID partitions. The committed splits (`data/100`, `data_alpaca/100`, `data_wiz/100`, 100 client files each) show:

| Dataset | Records per client (min / median / max) | Total | Label field | Generator script |
|---|---|---|---|---|
| Dolly | 49 / 137 / 288 | 15,011 | `category` | `client_data_allocation.py` |
| Alpaca | 520 / 520 / 521 | 52,002 | none | none in repo |
| Wizard | 700 / 700 / 700 | 70,000 | none | none in repo |

- **Dolly:** `client_data_allocation.py:40-63` splits each `category` across clients with Dirichlet(α = 0.5) proportions, requiring at least 40 records per client (seed 42). Client 0 holds two categories. Unequal client sizes are consistent with this.
  - The script also holds out 10 records per category as a test file (lines 17-36), but the committed client files total 15,011, the full Dolly-15k. **Unresolved:** whether the committed split came from this script as written.
- **Alpaca and Wizard:** equal-size clients and no label field. These splits are **not** Dirichlet over a label or over quantities as committed. **Unresolved:** how they were produced (shuffle, sorting or chunking), and what the paper means by Dirichlet for them.
- Evaluation uses a separate MMLU subset (`mmlu_test_1444.jsonl`), not these splits.

## 4. Communication accounting — Partly answered (8× vs 10× unresolved)

- **What the code logs:** `fed_utils/model_aggregation.py:336-363` logs, each round, `total_rank` (rows of every `A` plus columns of every `B` across adapted modules) and `total_param` (all aggregated adapter entries). The global adapter is counted **once per round, not per client**.
- **Table 2 efficiency = `1 / (total_rank / 2)`**, which is one over the summed rank across adapted modules. Checks:
  - TinyLlama FedIT, homogeneous: 22 layers × 2 modules (q, v) × rank 16 = 704 → 1/704 = 14.20 × 10⁻⁴ ✓.
  - LLaMA-3.2-1B FedIT: 16 × 2 × 16 = 512 → 19.53 × 10⁻⁴ (Table 2 prints 19.50).
- **Table 3 MB** match `parameters × 10 clients × 2 bytes / 10⁶` (decimal MB):
  - TinyLlama q/v adapters at rank 16 have 2,252,800 parameters, so FedIT is 45.056 MB (Table 3: 45.05).
  - Appendix F instead states `parameters × 2 / 1024²`, which gives 4.30 MB per client or 42.97 MB for 10 clients. Table 3 does not use the Appendix F formula.
- **FLoRA, 8× vs 10× (Unresolved):**
  - Table 3 has FLoRA at 360.45 / 45.05 = 8.0× FedIT's download.
  - Table 2's FLoRA efficiencies imply a stacked rank of about 128 = 8 × 16 per module (TinyLlama 1.78 → 5,618 / 44 modules; LLaMA 2.44 → 4,098 / 32 modules).
  - The Table 6 formulas, and the code's FLoRA stacking (`model_aggregation.py:239-272`, which is correctly structured), give 10 × 16 = 160 for 10 clients.
  - Nothing in the code explains the factor 8.

## 5. Layer shapes — Answered

- **LoRA target modules:** `main.py:65-68` defaults to `["q_proj", "v_proj"]`, and `run_wiz.sh` doesn't override it. Table 2's efficiencies match q and v only (item 4), as does Figure 6.
- **Discrepancy:** Appendix F says LoRA is applied to q, k, v and o.

| Model | Layers | Hidden | Heads / KV heads | q_proj | k_proj, v_proj | o_proj | Source |
|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 22 | 2048 | 32 / 4 | 2048 × 2048 | 256 × 2048 | 2048 × 2048 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` config.json |
| LLaMA-3.2-1B | 16 | 2048 | 32 / 8 (head dim 64) | 2048 × 2048 | 512 × 2048 | 2048 × 2048 | `unsloth/Llama-3.2-1B` config.json (public mirror; the Meta repo is gated) |

Shapes are written as `out × in` (PyTorch `nn.Linear.weight`), so in the paper's notation `ΔW ∈ ℝ^{m×n}`, m is the output size and n is the input size.

## 6. Stacking bug in the official aggregation — Answered

- **Where:** `fed_utils/model_aggregation.py:43-75`. The `else:` on line 56 follows the `for` loop on line 45, so Python parses it as `for … else`, which runs after every loop. I confirmed this with Python's `ast` module.
- **Effects:**
  - Client 0 is stacked twice: `B_stack = [B_0, B_0, B_1, …]`, `A_stack = [w_0A_0; w_0²A_0; w_1A_1; …]`. The aggregate becomes `ΔW + w_0² B_0A_0`.
  - With heterogeneous ranks, the whole stack so far is multiplied by `w_k` whenever its current rank equals client `k`'s rank (lines 45-55).
- **Replica:** a line-for-line NumPy copy, run on random adapters (m = 40, n = 30, equal weights), gave these errors relative to the correct `ΔW`:

  | Clients and ranks | Relative error |
  |---|---|
  | 10 clients, rank 16 | 3.1% |
  | 3 clients, rank 4 | 20% |
  | ranks 4, 8, 16 | 48% |

- **Not affected:** the SVD, `P`, threshold and global-adapter code (lines 80-115) matches Eqs. 1-3 and 6. If τ is so close to 1 that rounding means no p reaches it, the code keeps the full rank.
- **For the app:** implement exact weighted stacking (Algorithm 1). The effect on the paper's reported results is unknown and must not be claimed; real client updates are correlated, unlike these random examples.

## 7. LoRA scaling — Answered

The adapter output is scaled by `α / r`:

| Setting | Configuration | Scale |
|---|---|---|
| Homogeneous | `lora_r = 16`, `lora_alpha = 32` (`main.py:62-63`) | 2 |
| Heterogeneous | `lora_alpha = 2·r_k` (`main.py:386-394`) | 2 |

The scale is the same for every client, so the server's aggregation (which ignores it) is consistent. The toy experiment uses the same rule.

## 8. Heterogeneous rank distribution — Answered

`main.py:129-141` builds the client ranks from `{64: 10, 32: 10, 16: 20, 8: 20, 4: 40}`, matching Sec. 4.1. The `local_ranks` default argument (`main.py:81`) is overwritten each round (`main.py:307`).

## 9. Internal inconsistencies in the paper — Answered

| Topic | One place | Another place |
|---|---|---|
| MMLU peak threshold (TinyLlama, Wizard) | `τ = 0.91` (Figure 7) | `τ = 0.92` (Sec. 4.5 text) |
| FLoRIST server cost | 466.95M FLOPs (Table 4) | "466.9 FLOPs" (Sec. 4.4 text) |
| Communication efficiency | `1 / total download rank` (Sec. 4.1) | `1 / (total rank / 2)` (Appendix F); Table 2 uses the latter |
| MB conversion | `× 2 / 1024²` (Appendix F) | Table 3 matches `× 2 / 10⁶ × 10 clients` |
| LoRA target modules | q, k, v, o (Appendix F) | q, v (code default, Table 2, Figure 6) |
| Round-1 `A` initialization | `N(0, σ²)` (Algorithm 1) | Kaiming-uniform (PEFT default in code) |
| Reduction vs Full FT | "400×" (Figure 4) | "up to 349×" (Appendix F.1) and "403×" (Appendix B) |

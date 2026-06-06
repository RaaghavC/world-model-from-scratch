# World Models & Spatial Intelligence — Frontier Briefing (June 5, 2026)

Fresh, adversarially fact-checked survey (deep-research harness: 6 search angles,
28 sources fetched, 138 claims extracted, 25 verified by 3-vote, 23 confirmed).
Complements the broader `docs/RESEARCH.md`; this file captures the **2025–2026
frontier** that `wm2` is designed against. Every claim below survived 3-0 or 2-1
verification unless flagged.

## The field has bifurcated into two paradigms

1. **Generative video world models** — synthesize navigable environments
   *frame-by-frame, autoregressively*, conditioning each new frame on the growing
   trajectory. No explicit 3D (NeRF/Gaussian splatting deliberately rejected).
2. **JEPA-style predictive models** — forecast in a *learned latent space* (no
   pixels) for understanding and robot control.

`wm2` builds a small instance of **both** on one toy world (the RSSM generative
core + the decoder-free JEPA track), so the central 2026 debate is reproduced and
measured locally.

## Generative frontier (verified)

- **Genie 3** (DeepMind, flagship): real-time **720p @ 24 fps**, consistent for
  *several minutes*, **visual memory ~1 minute** back (Genie 2 was 10–20 s).
  Frame-by-frame autoregressive; **no explicit 3D**. "Project Genie" rolled out to
  AI Ultra users **Jan 29 2026**. [deepmind.google/blog/genie-3-a-new-frontier-for-world-models]
- **Decart MirageLSD**: real-time **causal autoregressive diffusion**, each frame
  in **<40 ms** (24 fps), conditioned on a window of past generated frames + the
  input frame + a prompt; claims "infinite"-length video. [decart.ai/publications/mirage]
- **Dominant generative architecture**: a **Diffusion Transformer (DiT)** over a
  **3D causal VAE** with **tubelet (spacetime-cube) patchification**. Streaming
  requires reformulating bidirectional diffusion into **causal** form (causal
  attention masks, per-frame/per-chunk noise schedules) + **forcing** strategies
  (Self-Forcing, Rolling Forcing) to fight exposure bias. (Survey arXiv:2603.28489;
  CogVideoX, NVIDIA Cosmos, MAGI-1, CausVid.)
- **Genie three-module recipe** (the canonical *interactive-control* blueprint,
  verified against Bruce et al. arXiv:2402.15391): **(1) spatiotemporal video
  tokenizer → (2) latent action model** (infers actions *unsupervised* between
  frame pairs) **→ (3) autoregressive dynamics model** (MaskGIT next-token).
  Enables frame-by-frame control **trained with no ground-truth action labels**.
  → `wm2/lam.py` is a small instance of module (2).
- **NVIDIA Cosmos 3** (tech report **June 3 2026**): a two-tower mixture-of-
  transformers unifying physical reasoning, world generation, and action
  generation. **Tencent HY-World 2.0** also active.

## Non-generative / JEPA frontier (verified)

- **V-JEPA 2** (Meta FAIR, arXiv:2506.09985): ViT-g, **>1B params**, **mask-
  denoising feature prediction** — predicts masked video in a *learned latent
  space*, L1 to a stop-grad EMA target encoder. No decoder, no pixels.
- **V-JEPA 2-AC**: freeze the encoder, train a **block-causal autoregressive
  predictor** on **~62 h** of unlabeled Droid robot video → **zero-shot robot
  manipulation** on Franka arms via **MPC (Cross-Entropy Method, receding
  horizon)**: 100% reach, ~65% grasp, 65–80% pick-and-place.
  → `wm2/jepa.py` + `wm2/plan.py` are a small instance of this recipe (EMA-target
  latent prediction + CEM planning).

## Spatial intelligence / persistent 3D (verified)

- **World Labs Marble** (Fei-Fei Li), GA **Nov 12 2025**: multimodal world model
  generating **persistent 3D worlds** from text/image/video/coarse-3D, exportable
  as **Gaussian splats / meshes / videos** (splats highest fidelity; meshes carry
  colliders for physics). The "persistent explicit-3D" counterpoint.
- **World Labs RTFM**: an autoregressive **diffusion transformer as a learned
  renderer** — novel viewpoints with **no explicit geometry**, only a weak
  3D-Euclidean prior for a KV-cache spatial memory.
- **3D/4D world modeling** uses native spatial reps beyond RGB — RGB-D, occupancy
  grids, LiDAR — over VAE/GAN/diffusion/AR backbones + NeRF/Gaussian-splat scene
  reps. (Survey arXiv:2509.07996.)

## Mechanistic vocabulary for builders (arXiv:2601.17067, Jan 2026)

A world model factors into **Observation** O_t (pixels; partial view), **State**
S_t (latent keeping task-relevant variables, filtering noise), and **Dynamics**
S_{t+1}=f(S_t,A_t); with two operations: **state estimation** (compress obs→latent)
and **state transition** (predict future state). State designs split into:

- **Implicit state** (manage raw history): compression (FramePack), retrieval
  (WorldMem), consolidation (StreamingT2V).
- **Explicit state** (compressed variables): **coupled** — recurrence *inside* the
  backbone (Mamba SSMs; **RAD = LSTM inside a DiT**); **decoupled** — a standalone
  transition model (LLM-semantic Owl-1; or geometry via render-generate-update on
  point clouds/Gaussians/meshes).

→ **`wm2`'s RSSM is an explicit, coupled state** (a GRU recurrence carrying memory
in the backbone) — the same family as RAD, and the modern successor to the
baseline's MDN-RNN.

## Long-horizon recipe (engineer-actionable, verified)

The drift that caps autoregressive length is fought with **Diffusion Forcing**
(independent per-frame noise levels; Chen et al., NeurIPS 2024) + **history
augmentation** (fine-tune on *deliberately corrupted* history frames so the model
learns to recover from its own errors). Lineage: Self-Forcing++, Rolling Forcing.

→ **`wm2` adopts a small-scale analogue**: *latent history augmentation* — during
world-model training, with probability `p` we replace the posterior latent fed to
the next step with the model's **own prior sample**, teaching it to recover from
its own rollout errors. This is the RSSM-native version of history augmentation,
and it directly attacks the long-horizon drift the baseline flagged.

## Open problems (3D/4D survey §6, the field's consensus list)

1. Standardized **benchmarking & evaluation** (WorldScore, Physics-IQ exist but no
   accepted leaderboard).  2. **High-fidelity, long-horizon** generation.
3. **Physical fidelity + controllability + generalization.**  4. **Real-time
   efficiency** (the <40 ms/frame budget).  5. **Cross-modal coherence.**

→ `wm2/evaluate.py` reports a small **WorldScore-style** triple
(**controllability, consistency, fidelity**) plus a **physics-faithfulness** probe
(does the dream conserve momentum/energy across collisions?) — directly targeting
problems 1–3 at toy scale.

## Caveats (from the verification pass)

Frontier figures (Genie 3, MirageLSD, Marble) originate from vendor blogs, re-
reported but rarely independently measured. MirageLSD's "first infinite" is
marketing (prior art exists). No benchmark *leaderboard numbers* (WorldScore /
physics scores) survived verification — only the benchmarks' existence. Coverage
gaps (not verified here): GameNGen, Veo-as-world-model, Cosmos specifics, Wayve
GAIA-2, Muse/WHAM, Runway, Odyssey. The DiT-over-3D-causal-VAE pattern is the
prevailing *generative* one, **not** universal — the JEPA line is a genuinely
competing paradigm.

## Key sources

Genie 3 (deepmind.google/blog/genie-3-a-new-frontier-for-world-models) ·
MirageLSD (decart.ai/publications/mirage) · V-JEPA 2 (arXiv:2506.09985) ·
Marble (worldlabs.ai/blog/marble-world-model) · RTFM (worldlabs.ai/blog/rtfm) ·
Genie (arXiv:2402.15391) · Efficient video WM survey (arXiv:2603.28489) ·
Mechanistic state/dynamics taxonomy (arXiv:2601.17067) · 3D/4D survey
(arXiv:2509.07996) · WorldScore (haoyi-duan.github.io/WorldScore) · Physics-IQ
(physics-iq.github.io) · Cosmos 3 (June 3 2026) · Tencent HY-World 2.0.

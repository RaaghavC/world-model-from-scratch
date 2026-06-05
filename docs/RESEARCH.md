# AI World Models: A Builder's Briefing (June 2026)

## Executive Summary

A **world model** is a machine learning system that builds an internal representation of an environment and predicts how that environment changes over time in response to actions ([Wikipedia](https://en.wikipedia.org/wiki/World_model_(artificial_intelligence))). The idea is older than deep learning: Kenneth Craik argued in 1943 that the mind runs a "small-scale model" of external reality, and reinforcement learning later formalized the setting as a **partially observable Markov decision process (POMDP)** — an agent takes actions that change a hidden world *state*, but never sees that state directly, only partial *observations* ([World Labs](https://www.worldlabs.ai/blog/taxonomy-of-world-models)).

The reason 2024–2026 is the inflection point is that four previously separate research communities — reinforcement learning, computer vision, robotics, and generative AI — converged on building these systems at once, each with its own meaning. By June 2026, "world model" has become, in Fei-Fei Li's words, "one of the most important and most overloaded terms in AI" ([World Labs](https://www.worldlabs.ai/blog/taxonomy-of-world-models)). The concrete triggers: DeepMind's Genie line went from ~1 frame per second (Genie 1, Feb 2024) to real-time 720p/24fps interactive worlds (Genie 3, Aug 2025), shipping to consumers as "Project Genie" on January 29, 2026 ([Google](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/)); OpenAI's Sora report reframed video generation as "world simulation" (Feb 2024); DreamerV3 was published in *Nature* (April 2, 2025) as the first single algorithm to master 150+ tasks and mine Minecraft diamonds from scratch; NVIDIA shipped Cosmos as a "World Foundation Model" platform for Physical AI (Jan 2025); and Fei-Fei Li's World Labs released Marble and crystallized the agenda of "spatial intelligence."

This briefing is written for someone about to **build** a world model from scratch in the Ha & Schmidhuber V+M+C lineage — a convolutional VAE, a mixture-density RNN, and an evolved controller that learns inside its own dream. We open with the conceptual map (Fei-Fei Li's functional taxonomy), then go deep on the foundational architectures you will actually implement, then survey the frontier so you understand where the simple thing you build sits in the larger landscape.

## Fei-Fei Li's Functional Taxonomy

Because the term is so overloaded, the most useful organizing frame available in 2026 is the **functional taxonomy** introduced in *A Functional Taxonomy of World Models: Renderers, Simulators, Planners, and the Loop That Connects Them*, published June 3, 2026 by Dr. Fei-Fei Li with the World Labs team ([Substack](https://drfeifei.substack.com/p/a-functional-taxonomy-of-world-models); [World Labs](https://www.worldlabs.ai/blog/taxonomy-of-world-models)). Both the title/subtitle and the June 3, 2026 date are verified against the primary sources and corroborated by independent third-party coverage.

The taxonomy grounds everything in the POMDP loop and then classifies world models by **what they output**:

- **Renderers** output *observations* — "pixels meant for human eyes, and the quality that matters most is visual fidelity." Named examples: Google **Genie 3**, World Labs **RTFM**, and Google **Nano Banana**.
- **Simulators** output *state* — "a geometrically, physically or dynamically faithful representation of the world that humans and computer programs can both compute on and interact with." Simulators demand accuracy beyond visual plausibility: geometry that holds under inspection, physics that respects Newton's laws. Named examples: World Labs **Marble** and **NVIDIA Omniverse**.
- **Planners** output *actions* — "Given an observation and a goal, a planner answers the question of what the agent should do next." This category includes **Vision-Language-Action (VLA)** models and **World Action Models**.

The article's central argument is that these are not three competing camps but **"three projections of a single underlying understanding."** A model that truly understands how a cup sits on a table "should be able to render that cup from any angle, simulate what happens when the cup is pushed, and plan for a hand to pick the cup up." Simulators are singled out as the **"bridge"** and **"linchpin,"** because "geometry, physics, and dynamics are the world itself" ([World Labs](https://www.worldlabs.ai/blog/taxonomy-of-world-models)).

The taxonomy doubles as a critique of the dominant video-generation paradigm: a renderer can be beautiful but "cannot be trusted to design a building or train a robot" — it may produce physically impossible flames — whereas a simulator serves both human eyes *and* computational agents like robots and RL systems ([Substack](https://drfeifei.substack.com/p/a-functional-taxonomy-of-world-models)).

This builds on Li's earlier manifesto, *From Words to Worlds: Spatial Intelligence is AI's Next Frontier* (November 10, 2025), which defines world models as "a new type of generative models" whose understanding, reasoning, generation, and interaction with "semantically, physically, geometrically and dynamically complex worlds" are "far beyond the reach of today's LLMs," which "remain wordsmiths in the dark; eloquent but inexperienced, knowledgeable but ungrounded" ([Substack](https://drfeifei.substack.com/p/from-words-to-worlds-spatial-intelligence)). She lists three essential capabilities: **generative** (worlds with perceptual, geometric, and physical consistency), **multimodal** (images, video, depth, text, gestures, actions as input), and **interactive** (output the next state given an action). World Labs was co-founded by Justin Johnson, Christoph Lassner, Ben Mildenhall, and Fei-Fei Li.

**A note for builders.** The taxonomy is one axis of cuts; the academic literature uses a different one. The survey *Understanding World or Predicting Future?* (arXiv [2411.14499](https://arxiv.org/html/2411.14499v1)) splits the field between **"understanding the world"** (building internal representations of mechanisms — the Ha/Schmidhuber lineage) and **"predicting the future"** (simulating future states to guide decisions — LeCun's emphasis). Whether the output-based taxonomy and the understand-vs-predict axis are reconcilable or orthogonal remains an open question. For the VAE+MDN-RNN system in this briefing, you are building a **renderer** (the VAE decodes pixels) wired to a **dynamics model** that you will use both as a simulator-of-latents and as the substrate for a **planner** (the evolved controller) — which is precisely the "three projections of one understanding" idea, in miniature.

## Foundational Architectures (the lineage you are building)

This is the section to implement from. The lineage runs Schmidhuber (1990) → Ha & Schmidhuber (2018) → PlaNet (2018) → Dreamer (2020).

### The C–M framework (Schmidhuber, 1990)

The conceptual seed is Schmidhuber's *Making the World Differentiable* (FKI-126-90, TUM, 1990), which proposed using a fully recurrent, self-supervised network as a **differentiable model of the environment** that predicts sensory inputs and rewards, with a separate **controller** that plans inside it ([thread](https://x.com/SchmidhuberAI/status/1991605057989521758)). The "model + controller" split — heavy world model, light policy — is the architectural DNA of everything below.

### Ha & Schmidhuber's *World Models* (2018)

The canonical operationalization is *World Models* (arXiv [1803.10122](https://arxiv.org/abs/1803.10122); the NeurIPS 2018 oral version is titled *Recurrent World Models Facilitate Policy Evolution*, arXiv [1809.01999](https://arxiv.org/abs/1809.01999)). The agent has **three explicitly separated components**, verified verbatim against the authors' interactive paper ([worldmodels.github.io](https://worldmodels.github.io/)):

> "Our agent consists of three components that work closely together: Vision (V), Memory (M), and Controller (C)."

**V — Vision (a Variational Autoencoder).** A ConvVAE encodes each 64×64×3 RGB frame into a low-dimensional latent vector *z* using **4 stride-2 convolutional layers** to μ and σ, and **4 deconvolutional layers** to reconstruct, trained for **1 epoch** on random-policy data with an **L2 reconstruction + KL loss**. The latent dimension *Nz* is **32 for CarRacing** and **64 for VizDoom**. Every one of these details is confirmed in the paper's Appendix A.1 (verdict: *supported*). Component size is ~4.35M parameters.

**M — Memory (an MDN-RNN).** An LSTM combined with a Mixture Density Network models the *probability density* of the next latent: P(z_{t+1} | a_t, z_t, h_t), as a **mixture of 5 diagonal-covariance Gaussians**. The LSTM has **256 hidden units** for CarRacing and **512** for VizDoom — a recurrent version of Bishop's (1995) mixture-density network. Modeling a distribution (not a point) is the crucial design choice: it captures stochasticity and lets you *sample* plausible futures, i.e. dream. M is ~422K parameters (CarRacing) / ~1.68M (VizDoom).

**C — Controller (a tiny linear policy).** C is a single linear layer with a tanh nonlinearity, a_t = W_c·[z_t, h_t] + b_c, with only **867 parameters** (CarRacing) or **1088** (VizDoom). It is deliberately small so it can be optimized by **CMA-ES** (population 64, each candidate averaged over 16 rollouts), which works well for solution spaces of a few thousand parameters. This deliberate asymmetry — virtually all capacity in the world model, almost none in the policy — is the defining feature of the architecture.

**Results.** On CarRacing-v0 the agent scored **906 ± 21** over 100 trials (the solving threshold is 900), beating prior Deep RL baselines (DQN 343, A3C-continuous 591, A3C-discrete 652) and the previous leaderboard best of 838 ± 11.

**Training inside the dream.** On VizDoom Take Cover, the controller was trained *entirely inside the dream* — M used as a simulator predicting both z_{t+1} and the done flag, never touching the game engine during policy optimization — then transferred to ~1100 real timesteps (750 "solves"). The key subtlety is the **MDN-RNN temperature τ**, which controls dream stochasticity and prevents the controller from exploiting the model's adversarial flaws. At τ=0.10 the agent scored **2086 in-dream but only 193 in reality**; at τ=1.15 it scored 918 in-dream and a best real transfer of **1092**. A noisier dream is a harder dream, and it closes the reality gap.

This maps directly onto a from-scratch build: a ConvVAE (4 down, 4 up, z_dim≈32), an MDN-RNN (256 LSTM units, K=5 Gaussians, NLL loss `−logsumexp(logπ + log N)`), and a CMA-ES controller scored on dream rollouts with a tunable temperature. One genuinely open question from the reproductions (Tallec et al.) is whether **training M matters at all**, or whether a random RNN feature extractor suffices for CarRacing — worth checking empirically. Two other open issues: temperature tuning is task-specific and hand-tuned rather than principled, and the linear CMA-ES controller scales poorly to high-dimensional actions or long-horizon credit assignment.

### PlaNet → the Recurrent State-Space Model (2018)

*Learning Latent Dynamics for Planning from Pixels* (PlaNet, arXiv [1811.04551](https://arxiv.org/abs/1811.04551), Google Brain/DeepMind) introduced the architecture that dominates model-based RL: the **Recurrent State-Space Model (RSSM)**. Its insight is to split the latent state into two paths:

- a **deterministic** recurrent path, h_t = f(h_{t−1}, s_{t−1}, a_{t−1}), implemented as a **GRU (200 units)**, and
- a **stochastic** path with prior p(s_t | h_t), plus an observation model p(o_t | h_t, s_t), a reward model p(r_t | h_t, s_t), and a posterior/encoder q(s_t | h_t, o_t).

The authors describe it as "a non-linear Kalman filter or sequential VAE." Crucially, **ablations show both paths are necessary**: a purely deterministic RNN cannot represent multiple plausible futures, and a purely stochastic state-space model cannot reliably retain information over many timesteps ("the agent does not learn without" the deterministic component). PlaNet also introduced **latent overshooting**, a training objective generalizing the one-step ELBO to enforce consistency of multi-step latent predictions via KL terms, improving long-horizon accuracy without decoding images at every step.

PlaNet has **no learned policy**. It plans online with the **Cross-Entropy Method (CEM)** for Model Predictive Control: horizon H=12, 1000 candidate action sequences per iteration, top-100 refit, 10 iterations, all evaluated in latent space. It matched D4PG using roughly **one-fiftieth the episodes** on six DeepMind Control Suite tasks, training in 10–20 hours on a single V100.

### Dreamer → behaviors by latent imagination (2020)

*Dream to Control* (Dreamer, arXiv [1912.01603](https://arxiv.org/abs/1912.01603), ICLR 2020; "DreamerV1" is a later retronym) reuses PlaNet's RSSM but **replaces online planning with a learned actor-critic trained in imagination**. Verified against the paper and the author's reference code: distributions in latent space are **30-dimensional diagonal Gaussians** (the stochastic part of the RSSM state); the **actor** is a **tanh-transformed Gaussian** policy; behavior is learned over an **imagination horizon H=15** using **λ-returns** (γ=0.99, λ=0.95). The defining mechanism is that "the action model uses analytic gradients through the learned dynamics to maximize the value estimates" — explicitly contrasted with Reinforce-style gradients (A3C, PPO). Dreamer reached **823 average across 20 DMC tasks at 5×10⁶ steps**, beating D4PG (786 at 1×10⁸ steps) and PlaNet (332), while training ~3 h per 1e6 steps versus PlaNet's 11 h.

The lineage thus offers **three ways to use a learned latent world model for control**: evolve a tiny controller inside MDN-RNN dreams (Ha & Schmidhuber); do derivative-free online CEM planning every step with no learned policy (PlaNet); or learn an actor-critic by differentiating value estimates through the RSSM (Dreamer). For a first build, the Ha & Schmidhuber route is the simplest and most legible.

## The Dreamer Line and Model-Based RL in Imagination

Dreamer matured into a family that established **learning behaviors purely inside a learned latent world model**, eliminating online environment interaction during policy optimization.

**DreamerV2** — *Mastering Atari with Discrete World Models* (arXiv [2010.02193](https://arxiv.org/abs/2010.02193), ICLR 2021) — made one decisive change: the RSSM stochastic state became **32 categorical variables × 32 classes**, trained with straight-through gradients, plus **KL balancing (α=0.8)**. It was the **first agent to reach human-level on Atari-55** by learning purely in its world model, with a **gamer-normalized median of 2.15**, beating Rainbow (1.47) and IQN (1.29) at 200M frames on a single V100 in under 10 days. The move from PlaNet's single continuous Gaussian latent to discrete categorical latents is itself an open research question — why discrete latents work better for world modeling is not fully settled.

**DreamerV3** — *Mastering Diverse Domains through World Models* (arXiv [2301.04104](https://arxiv.org/abs/2301.04104), Jan 2023), published in *Nature* April 2, 2025 ([Nature](https://www.nature.com/articles/s41586-025-08744-2)) — is a **single fixed-hyperparameter algorithm** spanning 150+ tasks across 8 domains (Atari, ProcGen, DMLab, Atari100k, Proprio/Visual Control, BSuite, and more). It comprises a world model (RSSM), a critic, and an actor that learn purely from latent trajectories over a planning horizon of 16. Its cross-domain robustness comes from a now-standard toolkit: **symlog** transforms `symlog(x)=sign(x)·ln(|x|+1)`; **two-hot symexp** distributional reward/value targets (255 buckets); **free bits** clipping KL below 1 nat; **percentile return normalization** (95th–5th, clamped to ≥1) so sparse near-zero rewards are not amplified; and a fixed entropy scale. It was the **first to collect Minecraft diamonds from scratch** with no human data or curricula — all agents find diamonds within 100M environment steps (first diamond after ~29–30M steps on average, roughly 17 in-game days). It shows **clean scaling**: larger models (the arXiv XS–XL family ~8M–200M; the *Nature* version lists 12M–400M) achieve both higher final scores *and* greater data efficiency. Benchmark highlights: Atari100k 49% (vs IRIS 29%), Atari-200M median 302% (vs DreamerV2 219%), and on DMLab 54.2% at 50M steps vs IMPALA's 51.3% at 10B — a ~200× data-efficiency gain. (The exact parameter ladder differs between the arXiv and *Nature* versions; treat the *Nature* methods section as canonical.)

**DayDreamer** — *World Models for Physical Robot Learning* (arXiv [2206.14176](https://arxiv.org/abs/2206.14176), CoRL 2022) — took the DreamerV2 recipe to **4 real robots with no simulator**, using an asynchronous actor process and learner process on a single GPU with shared hyperparameters. A **Unitree A1 quadruped learned to roll over, stand, and walk from scratch in ~1 hour** (and within ~10 more minutes adapted to withstand pushes), where a SAC baseline could not stand or walk. Two arms learned visual pick-and-place from sparse rewards: a UR5 reached ~2.5 objects/min after ~8 h and an XArm ~3.1 objects/min after ~10 h, while PPO and Rainbow failed on sample-efficiency. A Sphero wheeled robot learned visual navigation in ~2 h. (The arm/Sphero figures come from the project page and a secondary writeup; the PMLR v205 body rendered as image-only and should be cross-checked.)

**TD-MPC / TD-MPC2** form a parallel, *decoder-free* line. *Temporal Difference Learning for MPC* (TD-MPC, arXiv [2203.04955](https://arxiv.org/abs/2203.04955), ICML 2022) learns a Task-Oriented Latent Dynamics model plus a TD-learned terminal value, then plans short-horizon **MPPI** trajectories in latent space — solving Humanoid and Dog locomotion in 1M steps. *TD-MPC2* (arXiv [2310.16828](https://arxiv.org/abs/2310.16828), ICLR 2024 Spotlight) uses an **implicit world model trained by joint-embedding latent consistency with no reconstruction**, reward/value via discrete regression in log space, and SimNorm normalization. One configuration spans **104 continuous-control tasks** across 4 domains, and a single **317M-parameter agent performs 80 tasks** across embodiments, with normalized score scaling roughly linearly in log-parameters from 1M to 317M. TD-MPC2 directly probes the question your build implicitly raises: **is decoding pixels necessary at all?** It suggests reward/value-consistency objectives can replace reconstruction — at the cost of test-time planning compute, versus Dreamer's amortized, plan-free policy.

## Generative Interactive Environments (Neural Game Engines)

A "neural game engine" synthesizes playable video frame-by-frame, conditioned on user actions, with no underlying engine. **DeepMind's Genie line** defines this frontier.

**Genie 1** (arXiv [2402.15391](https://arxiv.org/abs/2402.15391), Feb 2024, lead author Jake Bruce) is an **11B-parameter foundation world model** with three components — a spatiotemporal (ST-transformer) video tokenizer, a **latent action model** (learning actions unsupervised), and an autoregressive dynamics model — trained on **200,000+ hours of unlabelled internet gaming video** with no action labels. The architecture and headline data figure are *supported* against the paper, with one verified nuance: the released 11B model was actually trained on a quality-filtered ~30,000-hour subset, while 200,000+ hours is the paper's own abstract framing of the collected pool. Genie 1 ran at only **~1 FPS**, which Wikipedia notes made the worlds "effectively unplayable as games." (The announcement is widely dated March 2024 per Wikipedia, but the arXiv v1 submission is Feb 23, 2024 — verdict: *partially-supported*; the actual unveiling was February 2024.)

**Genie 2** (Dec 4, 2024) is an **autoregressive latent-diffusion + causal-transformer** dynamics model prompted from a single image (often from Imagen 3), at 360p, with keyboard/mouse control and classifier-free guidance. It stays consistent for "up to a minute," though most shown examples last **10–20 seconds**, and shows emergent physics, lighting, and memory of out-of-view regions ([DeepMind](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/)).

**Genie 3** (Aug 5, 2025) is the **first real-time-playable Genie**: **720p at 24 FPS** (~41 ms/frame), several minutes of interaction, and **visual memory extending ~1 minute back** ([DeepMind](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)). It adds **"promptable world events"** (changing weather or adding objects via text mid-interaction) to expand counterfactual scenarios for agent training, and was demonstrated with DeepMind's **SIMA** embodied agent. DeepMind calls world models "a key stepping stone on the path to AGI." Its stated limits: a constrained action space, weak multi-agent interaction, imperfect geographic accuracy, poor in-world text, and no multi-hour sessions. It launched publicly as **"Project Genie" on January 29, 2026** to US AI Ultra subscribers (powered by Genie 3, Nano Banana Pro, and Gemini), but **capped at 60 seconds and with promptable world events excluded** ([Google](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/)).

Beyond Genie, several systems show the design space:

- **GameNGen** — *Diffusion Models Are Real-Time Game Engines* (arXiv [2408.14837](https://arxiv.org/abs/2408.14837), ICLR 2025) — repurposes **Stable Diffusion v1.4** to simulate DOOM at **>20 FPS on a single TPU**, with **PSNR 29.4**; human raters were only slightly better than chance at distinguishing it from real gameplay after 5 minutes. It is trained in two phases (record an RL agent, then train a diffusion model to predict the next frame from past frames + actions), and fights autoregressive drift by **adding Gaussian noise to encoded context frames**, plus decoder fine-tuning.
- **Oasis** (Decart + Etched, Oct 31, 2024) is the first open-source real-time playable world model — a **ViT autoencoder + DiT** Minecraft model at **20 FPS**, using **Diffusion Forcing + dynamic noising** for stability, with a 500M checkpoint open-sourced and optimized for Etched's Sohu ASIC ([oasis-model.github.io](https://oasis-model.github.io/)).
- **DIAMOND** (arXiv [2405.12399](https://arxiv.org/abs/2405.12399), NeurIPS 2024 Spotlight) is a diffusion world model built on the **EDM formulation with only 3 denoising steps**, motivated by the argument that compressing into discrete latents discards visual detail important for RL. It scores **1.46 mean human-normalized on Atari 100k**, beating IRIS (1.046) and DreamerV3 (1.097), with a ~4M-param U-Net; scaled to 381M it becomes a CS:GO neural engine at 10 Hz on an RTX 3090.
- **Microsoft Muse/WHAM** (Nature, Feb 2025) is a 1.6B World and Human Action Model trained on ~7 years of *Bleeding Edge* gameplay; the **WHAMM** MaskGIT variant (Apr 2025) emits all tokens for a frame at once to reach **10+ FPS at 640×360** ([Microsoft](https://www.microsoft.com/en-us/research/articles/whamm-real-time-world-modelling-of-interactive-environments/)).

The recurring engineering challenge — directly relevant to anyone dreaming with an MDN-RNN — is the **latency↔consistency tradeoff**: autoregressive rollouts suffer exposure bias and distribution drift, compounding errors over long horizons. Every system above attacks it differently (noise-augmented context, Diffusion Forcing, explicit memory). Newer work like BAgger (arXiv [2512.12080](https://arxiv.org/pdf/2512.12080), Dec 2025) teaches models to recover from their own rollout mistakes. This is the same phenomenon the temperature τ controls in the Ha & Schmidhuber dream.

## Video Generation as World Simulation — and the Physics Debate

The most contested claim in the field is whether **scaling video generators yields world models**. The thesis was stated in OpenAI's Feb 2024 report, literally titled *Video generation models as world simulators*: "scaling video generation models is a promising path towards building general purpose simulators of the physical world" ([OpenAI](https://openai.com/index/video-generation-models-as-world-simulators/)). Sora is a **diffusion transformer over spacetime latent patches**, generating up to 60s clips, and OpenAI claimed emergent 3D consistency, object permanence, world-interaction (a painter's strokes persist; bites leave marks), and even Minecraft control. In the *same* report OpenAI admitted Sora "does not accurately model the physics of many basic interactions, like glass shattering." The same "world model" framing recurs at Google (Veo 3), Runway (Gen-3/Gen-4's explicit "General World Models"), Luma (multimodal "worldsim"), and NVIDIA.

**The case against (scaling is not enough).** The central academic critique is ByteDance Seed's *How Far is Video Generation from World Model: A Physical Law Perspective* (arXiv [2411.02385](https://arxiv.org/abs/2411.02385), ICML 2025). Using a deterministic 2D simulator and scaling DiT models from 22.5M→456M parameters and data from 30K→3M videos, they found **near-perfect in-distribution but failing out-of-distribution** generalization — uniform-motion velocity error rose from 0.012 (ID) to 0.427 (OOD), ~35×. Mechanistically, the models generalize by **"case-based" retrieval** (mimicking the nearest training example) rather than abstracting rules, with a reference-matching priority of **color ≫ size ≫ velocity ≫ shape**. Their conclusion: "scaling alone is insufficient for video generation models to uncover fundamental physical laws." (Combinatorial generalization *did* improve with data diversity — abnormal rate fell from 67% to 10% as templates grew from 6 to 60.)

The benchmarks reinforce this. DeepMind's **Physics-IQ** (arXiv [2501.09038](https://arxiv.org/abs/2501.09038), Jan 2025) found **no correlation between visual realism and physical understanding** (Pearson r = −0.46): VideoPoet led on physics at 24.1%, while **Sora scored highest on visual realism (55.6%) but lowest on physics (8.7%)**. **VideoPhy** (arXiv [2406.03520](https://arxiv.org/abs/2406.03520), ICLR 2025) found the best model satisfied both caption adherence and physical commonsense for only **39.6%** of instances; **VideoPhy-2** (arXiv [2503.06800](https://arxiv.org/abs/2503.06800)) found the best model at **55.4%** joint score, dropping to 22% on the hard subset, with conservation laws (mass, momentum) the weakest area.

**The case for (it is emerging).** DeepMind's *Video models are zero-shot learners and reasoners* (arXiv [2509.20328](https://arxiv.org/abs/2509.20328), Sep 2025) argues Veo 3 is becoming a generalist vision foundation model via **"chain-of-frames"** reasoning, with large gains over Veo 2 in ~6 months: maze solving jumped to **78% pass@10 (vs Veo 2's 14%)**, object extraction to 93%. Sora 2 (Sep 2025) markedly improved physics (balls rebound rather than teleport). NVIDIA **Cosmos** (arXiv [2501.03575](https://arxiv.org/html/2501.03575v1), CES Jan 2025) sidesteps the debate by targeting **Physical AI** directly: a World Foundation Model platform with diffusion (7B/14B) and autoregressive (4B–13B) families plus tokenizers, defining a WFM as predicting the next observation from past observations and a perturbation, trained on ~20M hours of video (~9000T tokens) on 10,000 H100s; adopters include 1X, Agility, Figure, Skild, and Uber.

For a builder, the takeaway is concrete: **visual fidelity is not physical fidelity** — exactly Fei-Fei Li's renderer-vs-simulator distinction, now empirically demonstrated. If you want a model whose latents support *planning*, you should probe whether they encode dynamics (velocity, collisions), not just whether reconstructions look right.

## JEPA and the Self-Supervised, Non-Generative Bet

The most prominent dissent from pixel-prediction comes from **Yann LeCun**. His position paper *A Path Towards Autonomous Machine Intelligence* (2022) proposes a cognitive architecture of six modules — **Configurator, Perception, World Model, Cost, Actor, Short-term Memory** — all differentiable ([PDF](https://cis.temple.edu/tagit/presentations/A%20Path%20Towards%20Autonomous%20Machine%20Intelligence.pdf)). The world model is realized via **JEPA / Hierarchical-JEPA**, described as "a non-generative architecture for predictive world models that learn a hierarchy of representations." It **predicts in an abstract embedding space rather than reconstructing pixels**, uses latent variables to capture multiple plausible futures, and serves as "the basis of predictive world models for hierarchical planning under uncertainty" ([Meta](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)). LeCun explicitly bets *against* the generative/reconstruction approaches that the Ha & Schmidhuber VAE and the Sora/Genie lines embody.

This is the field's deepest unresolved fork, and it bears directly on your design choice. Your build reconstructs pixels (the VAE) and predicts latents with an MDN-RNN — generative on both counts. The JEPA thesis is that reconstruction wastes capacity modeling unpredictable detail and that abstract-space prediction is more scalable. TD-MPC2's decoder-free success is partial evidence for the JEPA side; the photoreal interactivity of Genie 3 and Marble is evidence for the generative side. **Whether LeCun's non-generative bet or the generative/interactive approach proves the more scalable substrate remains genuinely open** — it is one of the central debates of 2026, not a settled matter.

## Spatial Intelligence, 3D, and World Labs

Fei-Fei Li's World Labs anchors the **simulator/3D-native** wing of the taxonomy. **Marble** is described as the first world model promptable by multimodal inputs to **generate and maintain consistent 3D environments**, while **RTFM** (Real-Time Frame Model) is a real-time generative frame-based model using spatially-grounded frames ([Substack](https://drfeifei.substack.com/p/from-words-to-worlds-spatial-intelligence); [World Labs blog](https://www.worldlabs.ai/blog)). The thesis is that **"spatial intelligence"** — understanding and generating geometrically, physically, and dynamically coherent 3D worlds — is a distinct capability that pixel-only video models do not automatically acquire, and that simulators (faithful state) are the linchpin connecting renderers and planners.

This wing intersects the lineage you are building at the representation question: a VAE latent is a compressed 2D appearance code, whereas a "spatial" world model aims for a representation where geometry holds under inspection and physics is computable. Whether sufficiently scaled video/multimodal models acquire spatial intelligence implicitly, or whether 3D-native architectures are required, is listed as an explicit open question in the source material.

## Robotics, Embodied Agents, and Driving

World models reach the physical world along two tracks. The **model-based-RL track** is DayDreamer (above): a Dreamer-style latent world model learning real-robot behaviors online with no simulator, with the A1 quadruped walking in ~1 hour. The **planner track** is Fei-Fei Li's third category — **Vision-Language-Action (VLA) models and World Action Models** that output actions given observation and goal.

For **autonomous driving and Physical AI**, NVIDIA **Cosmos** is the flagship platform, and the source data spans driving (11%), hand motion (16%), human motion (10%), and spatial navigation (16%), explicitly aimed at training robots and AVs. Strikingly, **in February 2026 Waymo adopted Genie 3 to build a specialized "Waymo World Model" for autonomous-driving simulation** ([Wikipedia](https://en.wikipedia.org/wiki/Genie_(world_model))), a concrete instance of a generative interactive world model graduating from gaming into a safety-critical agent-training simulator. The caveat for builders: **how well these real-time world models actually transfer as robot/agent training simulators at scale is asserted more than demonstrated** — strong quantitative downstream results in primary sources remain thin.

## Evaluation and Benchmarks

Evaluation is the field's weakest joint, and this matters acutely for a builder who needs to know whether their model works. There is **no standardized, widely-adopted benchmark quantifying the latency–consistency–fidelity tradeoff** across interactive world models; cross-system comparisons remain largely qualitative, and WorldModelBench (CVPR 2025) is described as nascent.

For the *physics* question, three benchmark families now exist and broadly agree that current video models are weak: **Physics-IQ** (visual realism ⊥ physics, r = −0.46), **VideoPhy / VideoPhy-2** (best joint scores ~40–55%, worst on conservation laws), and the ByteDance physical-law probes (OOD failure, case-based generalization). For *model-based RL*, the field has mature, comparable benchmarks: Atari-55/Atari100k (gamer-normalized scores), the DeepMind Control Suite, DMLab, ProcGen, and Minecraft, with DreamerV3 and DIAMOND reporting head-to-head human-normalized numbers. For a from-scratch CarRacing/VizDoom-style build, the practical evaluation is the original one: **dream-vs-reality transfer score as a function of temperature**, plus latent probing (can a linear probe recover position/velocity from z?). A standardized way to measure a *simulator's* physical/geometric/dynamic faithfulness — as opposed to a renderer's plausibility — is asserted by the Fei-Fei Li taxonomy but **remains unsettled**.

## Limitations and Debates

Five fault lines run through the field as of June 2026, and a builder should hold all of them explicitly:

1. **Generative vs non-generative (LeCun's JEPA bet).** Should a world model reconstruct pixels or predict in abstract space? Unresolved; your build is squarely on the generative side.
2. **Scaling vs structure (the physics debate).** ByteDance shows scaling alone does not yield physical laws (case-based, not rule-based learning); DeepMind's Veo-3 results show rapid emergent gains. Both can be partly true.
3. **Renderer vs simulator fidelity.** The taxonomy's claim that a single model can be both a faithful simulator and a photoreal renderer, or whether the plausibility-vs-accuracy tradeoff forces specialization, is an open question — empirically supported by Physics-IQ's r = −0.46.
4. **Where the policy lives.** Ha/Schmidhuber and Dreamer fold prediction and control into one system; LeCun and the Fei-Fei Li taxonomy treat the planner as separate from the world model. This is a genuine architectural fork, not a cosmetic one.
5. **Drift and the reality gap.** Autoregressive rollouts compound errors; the dream is exploitable by the controller (the τ=0.1 → 193-in-reality result). Mitigations (temperature, noise augmentation, Diffusion Forcing) are effective but largely heuristic.

On provenance: the game-imitating models (DOOM, Minecraft, Bleeding Edge, CS:GO) raise **unresolved IP questions** — Oasis trains to reproduce Minecraft without its code or textures, and GameNGen reproduces DOOM. Anyone building on copyrighted gameplay should treat licensing as open.

## The 2025–2026 Frontier Landscape

The frontier is defined by three simultaneous pushes. **Real-time interactivity**: Genie 3 (720p/24fps, ~1-min memory) shipping as Project Genie (Jan 29, 2026), and WHAMM/Oasis hitting playable frame rates. **Scaling and unification**: DreamerV3's single-config mastery of 150+ tasks (*Nature*, Apr 2025), TD-MPC2's one-agent-80-tasks, and Cosmos's foundation-model platform for Physical AI. **The conceptual reckoning**: Fei-Fei Li's functional taxonomy (June 3, 2026) attempting to impose order on an overloaded term, alongside the still-live generative-vs-JEPA and scaling-vs-structure debates.

For someone building a VAE+MDN-RNN+controller system today, the situation is unusually favorable. You are implementing the **canonical, well-understood foundation** (Ha & Schmidhuber 2018) whose every hyperparameter is documented and verified, while the frontier above is essentially a sequence of upgrades to the *same three boxes*: swap the VAE for a diffusion renderer (DIAMOND, GameNGen) or a 3D-native simulator (Marble); swap the MDN-RNN for an RSSM with discrete latents (DreamerV3) or a decoder-free joint-embedding model (TD-MPC2) or a transformer (Genie); and swap the CMA-ES controller for a latent-imagination actor-critic (Dreamer) or MPPI planning (TD-MPC2). The lineage you are building is not a museum piece — it is the legible skeleton on which the entire 2026 frontier is hung.

## Bibliography

- A Functional Taxonomy of World Models (World Labs): https://www.worldlabs.ai/blog/taxonomy-of-world-models
- A Functional Taxonomy of World Models (Fei-Fei Li, Substack): https://drfeifei.substack.com/p/a-functional-taxonomy-of-world-models
- From Words to Worlds: Spatial Intelligence is AI's Next Frontier (Fei-Fei Li): https://drfeifei.substack.com/p/from-words-to-worlds-spatial-intelligence
- World Labs blog: https://www.worldlabs.ai/blog
- Building Spatially Intelligent AI (Radical Ventures): https://radical.vc/building-spatially-intelligent-ai/
- Time profile of Fei-Fei Li: https://time.com/7339693/fei-fei-li-ai/
- World model (artificial intelligence) — Wikipedia: https://en.wikipedia.org/wiki/World_model_(artificial_intelligence)
- Understanding World or Predicting Future? A Comprehensive Survey of World Models: https://arxiv.org/html/2411.14499v1
- Ha & Schmidhuber, World Models (arXiv): https://arxiv.org/abs/1803.10122
- Ha & Schmidhuber, World Models (interactive): https://worldmodels.github.io/
- Recurrent World Models Facilitate Policy Evolution (NeurIPS 2018): https://arxiv.org/abs/1809.01999
- Schmidhuber, Making the World Differentiable (FKI-126-90) reference: https://x.com/SchmidhuberAI/status/1991605057989521758
- PlaNet — Learning Latent Dynamics for Planning from Pixels: https://arxiv.org/abs/1811.04551
- PlaNet (ar5iv full text): https://ar5iv.labs.arxiv.org/html/1811.04551
- PlaNet project page: https://planetrl.github.io/
- Dreamer — Dream to Control (arXiv): https://arxiv.org/abs/1912.01603
- Dreamer (ar5iv full text): https://ar5iv.labs.arxiv.org/html/1912.01603
- DreamerV2 — Mastering Atari with Discrete World Models: https://arxiv.org/abs/2010.02193
- DreamerV3 — Mastering Diverse Domains through World Models (arXiv): https://arxiv.org/abs/2301.04104
- DreamerV3 — Mastering diverse control tasks through world models (Nature): https://www.nature.com/articles/s41586-025-08744-2
- DreamerV3 (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC12003158/
- DreamerV3 project page: https://danijar.com/project/dreamerv3/
- DayDreamer — World Models for Physical Robot Learning (arXiv): https://arxiv.org/abs/2206.14176
- DayDreamer project page: https://danijar.com/project/daydreamer/
- DayDreamer (PMLR v205): https://proceedings.mlr.press/v205/wu23c/wu23c.pdf
- TD-MPC: https://arxiv.org/abs/2203.04955
- TD-MPC2: https://arxiv.org/abs/2310.16828
- TD-MPC2 project page: https://www.tdmpc2.com/
- Genie: Generative Interactive Environments: https://arxiv.org/abs/2402.15391
- Genie (DeepMind publication page): https://deepmind.google/research/publications/60474/
- Genie 2 (DeepMind blog): https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/
- Genie 3 (DeepMind blog): https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/
- Project Genie (Google blog): https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/
- Genie (world model) — Wikipedia: https://en.wikipedia.org/wiki/Genie_(world_model)
- GameNGen — Diffusion Models Are Real-Time Game Engines (arXiv): https://arxiv.org/abs/2408.14837
- GameNGen project page: https://gamengen.github.io/
- DIAMOND (arXiv): https://arxiv.org/abs/2405.12399
- DIAMOND project page: https://diamond-wm.github.io/
- Oasis project page: https://oasis-model.github.io/
- Oasis (Hugging Face): https://huggingface.co/Etched/oasis-500m
- Microsoft Muse / WHAM (publication): https://www.microsoft.com/en-us/research/publication/world-and-human-action-models-towards-gameplay-ideation/
- WHAMM (Microsoft Research): https://www.microsoft.com/en-us/research/articles/whamm-real-time-world-modelling-of-interactive-environments/
- BAgger (Backwards Aggregation): https://arxiv.org/pdf/2512.12080
- OpenAI — Video generation models as world simulators: https://openai.com/index/video-generation-models-as-world-simulators/
- ByteDance — How Far is Video Generation from World Model (arXiv): https://arxiv.org/abs/2411.02385
- PhyWorld project page: https://phyworld.github.io/
- Physics-IQ (arXiv): https://arxiv.org/abs/2501.09038
- VideoPhy (arXiv): https://arxiv.org/abs/2406.03520
- VideoPhy-2 (arXiv): https://arxiv.org/abs/2503.06800
- Video models are zero-shot learners and reasoners (arXiv): https://arxiv.org/abs/2509.20328
- Video zero-shot project page: https://video-zero-shot.github.io/
- Google Veo 3 (CNBC): https://www.cnbc.com/2025/05/20/google-ai-video-generator-audio-veo-3.html
- Google Veo (DeepMind): https://deepmind.google/models/veo/
- Runway Gen-4: https://runwayml.com/research/introducing-runway-gen-4
- Runway Gen-3 Alpha: https://runwayml.com/research/introducing-gen-3-alpha
- Luma Ray2: https://lumalabs.ai/ray2
- NVIDIA Cosmos (arXiv): https://arxiv.org/html/2501.03575v1
- NVIDIA Cosmos World Foundation Models (blog): https://blogs.nvidia.com/blog/cosmos-world-foundation-models/
- LeCun — A Path Towards Autonomous Machine Intelligence (PDF): https://cis.temple.edu/tagit/presentations/A%20Path%20Towards%20Autonomous%20Machine%20Intelligence.pdf
- Meta — I-JEPA blog: https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/
- TuringPost — JEPA: https://www.turingpost.com/p/jepa

## Bibliography

- https://www.worldlabs.ai/blog/taxonomy-of-world-models
- https://drfeifei.substack.com/p/a-functional-taxonomy-of-world-models
- https://drfeifei.substack.com/p/from-words-to-worlds-spatial-intelligence
- https://www.worldlabs.ai/blog
- https://radical.vc/building-spatially-intelligent-ai/
- https://time.com/7339693/fei-fei-li-ai/
- https://en.wikipedia.org/wiki/World_model_(artificial_intelligence)
- https://arxiv.org/html/2411.14499v1
- https://arxiv.org/abs/1803.10122
- https://worldmodels.github.io/
- https://arxiv.org/abs/1809.01999
- https://x.com/SchmidhuberAI/status/1991605057989521758
- https://arxiv.org/abs/1811.04551
- https://ar5iv.labs.arxiv.org/html/1811.04551
- https://planetrl.github.io/
- https://arxiv.org/abs/1912.01603
- https://ar5iv.labs.arxiv.org/html/1912.01603
- https://arxiv.org/abs/2010.02193
- https://arxiv.org/abs/2301.04104
- https://www.nature.com/articles/s41586-025-08744-2
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12003158/
- https://danijar.com/project/dreamerv3/
- https://arxiv.org/abs/2206.14176
- https://danijar.com/project/daydreamer/
- https://proceedings.mlr.press/v205/wu23c/wu23c.pdf
- https://arxiv.org/abs/2203.04955
- https://arxiv.org/abs/2310.16828
- https://www.tdmpc2.com/
- https://arxiv.org/abs/2402.15391
- https://deepmind.google/research/publications/60474/
- https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/
- https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/
- https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/
- https://en.wikipedia.org/wiki/Genie_(world_model)
- https://arxiv.org/abs/2408.14837
- https://gamengen.github.io/
- https://arxiv.org/abs/2405.12399
- https://diamond-wm.github.io/
- https://oasis-model.github.io/
- https://huggingface.co/Etched/oasis-500m
- https://www.microsoft.com/en-us/research/publication/world-and-human-action-models-towards-gameplay-ideation/
- https://www.microsoft.com/en-us/research/articles/whamm-real-time-world-modelling-of-interactive-environments/
- https://arxiv.org/pdf/2512.12080
- https://openai.com/index/video-generation-models-as-world-simulators/
- https://arxiv.org/abs/2411.02385
- https://phyworld.github.io/
- https://arxiv.org/abs/2501.09038
- https://arxiv.org/abs/2406.03520
- https://arxiv.org/abs/2503.06800
- https://arxiv.org/abs/2509.20328
- https://video-zero-shot.github.io/
- https://www.cnbc.com/2025/05/20/google-ai-video-generator-audio-veo-3.html
- https://deepmind.google/models/veo/
- https://runwayml.com/research/introducing-runway-gen-4
- https://runwayml.com/research/introducing-gen-3-alpha
- https://lumalabs.ai/ray2
- https://arxiv.org/html/2501.03575v1
- https://blogs.nvidia.com/blog/cosmos-world-foundation-models/
- https://cis.temple.edu/tagit/presentations/A%20Path%20Towards%20Autonomous%20Machine%20Intelligence.pdf
- https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/
- https://www.turingpost.com/p/jepa
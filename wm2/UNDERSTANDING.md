# Understanding this project (from zero)

This guide assumes **no background** — no AI, no machine learning, no math beyond
high school. By the end you'll understand what a "world model" is, what this project
built, and why the results matter. Read it top to bottom; each idea builds on the
last.

---

## 1. The one-sentence idea

> A **world model** is a computer program that watches an environment, builds a
> little *mental model* of how it works, and can then **imagine** what will happen
> next — and even practice making decisions inside its own imagination before acting
> for real.

That's it. Everything below is detail.

A good human analogy: before you pour coffee, you can *picture* the cup filling and
stop before it overflows. You never actually overflowed a cup to learn that — you
ran a quick simulation in your head. A world model gives a computer that ability.

---

## 2. The setting: an agent in a tiny world

We made a tiny video-game world: a 64×64 pixel image with a small **orange dot**
(the "agent", which we control), a **green square** (the "goal", where we want the
agent to go), on a dark background. Every step, we can push the dot in one of 5 ways:
do nothing, or thrust **left / right / up / down**. The dot has *momentum* (like a
puck on ice) — it keeps gliding after you stop pushing.

The computer only ever sees the **raw picture** (the pixels). It is *never told*
where the dot is, how fast it's moving, or what the actions do. It has to figure all
of that out by itself, just from watching. That's the challenge.

---

## 3. Three jobs: See, Remember, Act (the "V-M-C" idea)

The original 2018 "World Models" paper (Ha & Schmidhuber) split the problem into
three parts. We keep this structure because it's the clearest way to think about it.

1. **See (Vision).** A 64×64 colour image is 12,288 numbers — too many to reason
   about. So we *compress* each picture into a short list of numbers (say, a few
   hundred) that captures "what matters." This short list is called a **latent**
   (think: a *summary* or *fingerprint* of the picture). A good summary throws away
   irrelevant detail and keeps the important stuff (where the dot is, where the goal
   is).

2. **Remember (Memory / the dynamics model).** Given the current summary and the
   action you take, predict the *next* summary. Do this over and over and you can
   roll the world forward **without looking at the real pixels** — that's
   **imagining**, or as the field calls it, **dreaming**. The "memory" part matters
   because a single picture doesn't show *velocity* (a photo of a moving dot looks
   the same as a photo of a still dot) — so the model must remember recent motion to
   predict where things go.

3. **Act (Controller / planner).** Use the model to choose actions that reach the
   goal. The striking trick from 2018: you can train the decision-maker **entirely
   inside the dream** — it practises in imagination and then works in the real world.

---

## 4. What's a "dream," concretely?

Take **one** real picture. Compress it to a summary. Now, *without ever looking at
the real game again*, repeatedly ask the Memory model "given this summary and this
action, what's the next summary?" — and feed each prediction back in. You get a whole
imagined trajectory. Decompress the summaries back into pictures and you can *watch*
the model's imagination. If the model learned the world well, its dream looks like
the real thing: the dot moves and drifts correctly, even though no real game is
running. (See `wm2_artifacts/dream_compare.gif` — left is real, middle is the
compress-then-decompress "reconstruction," right is the pure dream.)

Dreams aren't perfect: small errors **compound** the longer you imagine (like
photocopying a photocopy), so far-future dreams "drift." That's a real, known problem
in this field, and we measure it.

---

## 5. What's "new" in 2026 (and in this project)

The 2018 design still works, but the cutting edge replaced each of the three boxes
with something more powerful. We rebuilt all of them at small scale:

- **Discrete summaries (instead of smooth ones).** Modern models (DeepMind's
  "DreamerV3") summarize each picture using a set of *categorical choices* — like
  describing a scene by ticking boxes ("dot is upper-left," "moving right," ...)
  rather than with continuous dials. This turns out to be more robust. Our Memory
  model also predicts a *distribution* over next summaries (a *range* of plausible
  futures), which is what lets it dream.

- **Discovering the actions by itself (the "latent action model," from DeepMind's
  *Genie*).** Normally you tell the model "this was action #3." Genie's trick: show
  it only *pairs of consecutive frames* and let it *invent* its own action labels by
  noticing what changed. If it works, its invented labels line up with the real
  controls — controllability learned with **no labels at all**.

- **Predicting in "summary space" with no pictures (the "JEPA" idea, from Meta /
  Yann LeCun).** A competing philosophy says: don't bother reconstructing pixels at
  all; just predict future *summaries*. It's leaner and is a major 2026 research
  direction. We include a small version to compare the two philosophies.

- **Planning by trying things in imagination.** Instead of evolving a fixed
  decision-maker, at every moment we *imagine* the next few steps for each possible
  action, see which one gets closest to the goal, do that one, and repeat. This is
  "model-predictive control" — the same idea V-JEPA 2 uses to control real robots.

---

## 6. What we found (the results, in plain words)

We trained both the 2018 model and our 2026 model **from scratch on a laptop** and
compared them on the same task.

- **The summary is a genuine understanding of the world, not just a picture.** A
  tiny add-on can read the dot's **position** out of the summary almost perfectly
  (97% accurate), *and* its **velocity** (32%) — even though velocity is invisible in
  any single frame. The 2018 model never even checked for velocity; ours encodes it.
  This is the difference between *remembering what a scene looks like* and
  *understanding how it moves*.

- **The controller reaches the goal 97.5% of the time — having never touched the
  real game during training.** It practised purely in imagination. (Random flailing
  succeeds 25% of the time; the 2018 model got 39%.) The key lesson we learned the
  hard way: only imagine a *few* steps ahead before re-checking reality — imagine too
  far and the dream drifts and misleads you.

- **One thing only partly works, and we say so.** Getting the model to *discover the
  actions by itself* (the Genie trick) only partially succeeded here (it recovers
  about a third of the structure cleanly). We traced exactly why: with momentum, the
  effect of a push is tiny compared to the dot's existing glide, so the "discover the
  action" signal is faint. A supervised check shows the information *is* there (87%),
  but discovering it *without labels* is genuinely hard — which is exactly an open
  research problem at the real frontier, showing up in our toy.

The honest summary: **the modern model understands the world better and controls it
better, while inheriting the same hard problems the giant frontier models have — just
small enough to see clearly on a laptop.**

---

## 7. A mini-glossary

- **Agent**: the thing we control (the orange dot).
- **Environment / world**: the little game.
- **Pixels / frame**: one picture (a 64×64 grid of colours).
- **Latent / summary / embedding**: a short list of numbers that compresses a
  picture down to what matters.
- **Encoder / decoder**: the compressor (picture → summary) and decompressor
  (summary → picture).
- **Dynamics model**: predicts the next summary from the current one + an action.
- **Dream / imagination / rollout**: running the dynamics model forward on its own
  predictions, without looking at the real world.
- **Drift**: dreams getting less accurate the longer you imagine.
- **Controller / planner / policy**: the part that picks actions.
- **Probe**: a tiny add-on we train to *read* something (like position) out of the
  summary, to check what the summary "knows."
- **R² ("R-squared")**: a 0-to-1 score for how well a prediction matches the truth;
  1.0 is perfect, 0 is no better than guessing the average.
- **World model**: the whole See+Remember system that can imagine the world forward.

Next: [`USER_GUIDE.md`](USER_GUIDE.md) shows you how to run it yourself; the full
technical story is in [`README.md`](README.md), [`FINDINGS.md`](FINDINGS.md), and the
[paper](paper/paper.tex).

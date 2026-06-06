"""A from-scratch multi-body 2D physics world -- the "spatial" upgrade.

Where the baseline (`wm/env.py`) has a single thrusted ball, this world has
several bodies that move and **collide elastically with each other** under
optional gravity, are drawn with **depth-ordered occlusion** (a 2.5-D scene),
and expose a goal-reaching task for an agent body. Three properties make it a
much stronger test of a *world* model than the baseline:

  * **Physics that must be conserved.** Free bodies are frictionless and bounce
    elastically (restitution 1), so total momentum and kinetic energy are
    conserved across collisions -- a quantity a faithful "simulator" dream must
    preserve, and a renderer that merely looks plausible will not. This is the
    renderer-vs-simulator distinction (Fei-Fei Li, 2026) made measurable.
  * **Hidden velocity + occlusion.** A single frame never reveals velocity, and
    bodies pass in front of one another, so identity and motion must be carried
    in memory -- the same "visual memory" capability Genie 3 is celebrated for.
  * **Same interface as the baseline.** (img_size, img_size, 3) uint8 frames and
    5 discrete thrust actions, so the new model can be compared head-to-head.

Set `n_bodies=1, gravity=0` to recover a baseline-like single-ball task.

State (never shown to the model; used only for labels / reward / physics checks):
    pos  : (N,2) body centres in normalized [0,1]^2
    vel  : (N,2) body velocities (normalized units / step)
    z    : (N,)  pseudo-depth in [0,1] (render size + occlusion order)
    goal : (2,)  goal-square centre for the agent (body 0)
"""
from __future__ import annotations

import os

import numpy as np

from wm2.config import CFG, Config

# RGB palette. Background dark so bright bodies/goal stand out.
C_BG = (18, 18, 28)
C_WALL = (70, 70, 95)
C_GOAL = (40, 185, 80)
# Distinct body colours: body 0 (agent) is orange; others are cool hues.
BODY_COLORS = [
    (245, 130, 45),   # agent  (orange)
    (70, 140, 240),   # blue
    (210, 80, 200),   # magenta
    (230, 205, 70),   # yellow
    (90, 210, 200),   # teal
]

# Discrete action -> unit thrust direction (x, y). y points DOWN (image rows).
ACTION_VEC = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),
    2: (-1.0, 0.0),
    3: (0.0, 1.0),
    4: (0.0, -1.0),
}


class MultiBody2DEnv:
    """Deterministic multi-body thrust-and-bounce world with elastic collisions."""

    def __init__(self, cfg: Config = CFG, seed: int | None = None):
        self.cfg = cfg
        self.S = cfg.img_size
        self.N = cfg.n_bodies
        self.rng = np.random.default_rng(seed)
        ys, xs = np.mgrid[0 : self.S, 0 : self.S]
        self._xs = (xs + 0.5) / self.S
        self._ys = (ys + 0.5) / self.S
        # Per-body radius (agent vs free bodies) and mass (proportional to area).
        self.radii = np.array(
            [cfg.agent_radius] + [cfg.body_radius] * (self.N - 1), dtype=np.float64
        )
        self.mass = (self.radii ** 2)  # 2-D "mass" ~ area; only ratios matter
        self.pos = np.zeros((self.N, 2))
        self.vel = np.zeros((self.N, 2))
        self.depth = np.zeros(self.N)
        self.goal = np.zeros(2)
        self.t = 0
        self.reset()

    # ------------------------------------------------------------------ core MDP
    def reset(self, pos=None, vel=None, goal=None, depth=None) -> np.ndarray:
        """Start an episode with non-overlapping bodies and random free velocities."""
        c = self.cfg
        if pos is None:
            pos = self._sample_nonoverlapping()
        self.pos = np.array(pos, float).reshape(self.N, 2)
        # Free bodies (1..) get a random-direction velocity at a fixed speed; the
        # agent (0) starts at rest. Caller may override.
        if vel is None:
            vel = np.zeros((self.N, 2))
            for i in range(1, self.N):
                ang = self.rng.uniform(0, 2 * np.pi)
                vel[i] = c.free_speed0 * np.array([np.cos(ang), np.sin(ang)])
        self.vel = np.array(vel, float).reshape(self.N, 2)
        self.depth = (
            self.rng.uniform(0.2, 0.8, size=self.N) if depth is None
            else np.array(depth, float)
        )
        self.goal = (
            self.rng.uniform(0.18, 0.82, size=2) if goal is None
            else np.array(goal, float)
        )
        self.t = 0
        return self.render()

    def _sample_nonoverlapping(self) -> np.ndarray:
        """Rejection-sample N body centres that don't overlap each other."""
        pos = np.zeros((self.N, 2))
        for i in range(self.N):
            for _ in range(200):
                r = self.radii[i]
                p = self.rng.uniform(r, 1 - r, size=2)
                ok = all(
                    np.hypot(*(p - pos[j])) > (self.radii[i] + self.radii[j]) * 1.05
                    for j in range(i)
                )
                if ok:
                    pos[i] = p
                    break
            else:
                pos[i] = p
        return pos

    def step(self, a: int):
        """Advance one timestep under agent action `a`. Returns (obs, r, done, info)."""
        c = self.cfg
        # 1) Agent control. Default: thrust + friction + speed cap (momentum world,
        #    free bodies get no thrust/friction). direct_control: the action SETS the
        #    agent velocity directly (kinematic; actions drive the frames).
        ax, ay = ACTION_VEC[int(a)]
        if c.direct_control:
            self.vel[0] = np.array([ax, ay], float) * c.max_speed
        else:
            self.vel[0, 0] += ax * c.thrust
            self.vel[0, 1] += ay * c.thrust
            self.vel[0] *= c.friction
            sp = float(np.hypot(*self.vel[0]))
            if sp > c.max_speed:
                self.vel[0] *= c.max_speed / sp
        # 2) Gravity (applies to all bodies; default 0).
        if c.gravity:
            self.vel[:, 1] += c.gravity
        # 3) Integrate positions (explicit Euler).
        self.pos += self.vel
        # 4) Body-body elastic collisions (resolve approaching overlaps).
        self._resolve_body_collisions()
        # 5) Wall collisions: clamp inside [r, 1-r] and reflect with restitution.
        for i in range(self.N):
            r = self.radii[i]
            for d in range(2):
                if self.pos[i, d] < r:
                    self.pos[i, d] = r
                    self.vel[i, d] *= -c.restitution_wall
                elif self.pos[i, d] > 1 - r:
                    self.pos[i, d] = 1 - r
                    self.vel[i, d] *= -c.restitution_wall
        self.t += 1
        dist = float(np.hypot(*(self.pos[0] - self.goal)))
        reward = -dist
        done = self.t >= c.ep_len
        info = {"dist": dist, "reached": dist < c.goal_radius,
                "ke": float(self.kinetic_energy()), "mom": self.momentum().copy()}
        return self.render(), reward, done, info

    def _resolve_body_collisions(self) -> None:
        """Standard 2-D elastic collisions between overlapping circles.

        For each approaching, overlapping pair (i, j) we (a) push the bodies apart
        so they no longer interpenetrate and (b) exchange the velocity component
        along the collision normal using the mass-weighted elastic-collision law:

            v_i' = v_i - 2 m_j/(m_i+m_j) * <v_i-v_j, x_i-x_j>/|x_i-x_j|^2 (x_i-x_j)

        scaled by the restitution e (e=1 conserves momentum AND kinetic energy).
        """
        c = self.cfg
        e = c.restitution_body
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dp = self.pos[i] - self.pos[j]
                dist = float(np.hypot(*dp))
                rsum = self.radii[i] + self.radii[j]
                if dist >= rsum or dist < 1e-9:
                    continue
                n = dp / dist                       # collision normal (i <- j)
                # Only resolve if the bodies are approaching along the normal.
                dv = self.vel[i] - self.vel[j]
                vn = float(np.dot(dv, n))
                if vn < 0:                          # approaching
                    mi, mj = self.mass[i], self.mass[j]
                    # Symmetric elastic impulse with restitution.
                    jimp = -(1 + e) * vn / (1 / mi + 1 / mj)
                    self.vel[i] += (jimp / mi) * n
                    self.vel[j] -= (jimp / mj) * n
                # Positional correction: split the overlap along the normal.
                overlap = rsum - dist
                corr = (overlap / 2 + 1e-6) * n
                self.pos[i] += corr
                self.pos[j] -= corr

    # ------------------------------------------------------------- physics probes
    def kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.mass[:, None] * self.vel ** 2))

    def momentum(self) -> np.ndarray:
        return (self.mass[:, None] * self.vel).sum(0)

    # ------------------------------------------------------------------ rendering
    def render(self) -> np.ndarray:
        """Rasterize to (S,S,3) uint8, drawing far->near so nearer bodies occlude."""
        S = self.S
        img = np.empty((S, S, 3), np.uint8)
        img[:] = C_BG
        img[0, :] = img[-1, :] = C_WALL
        img[:, 0] = img[:, -1] = C_WALL
        # Goal square (drawn under the bodies).
        gr = self.cfg.goal_radius
        gmask = (np.abs(self._xs - self.goal[0]) < gr) & (
            np.abs(self._ys - self.goal[1]) < gr
        )
        img[gmask] = C_GOAL
        # Bodies: far (large depth) first so near bodies are painted last (on top).
        order = np.argsort(-self.depth) if self.cfg.depth_render else range(self.N)
        for i in order:
            r = self.radii[i]
            amask = (self._xs - self.pos[i, 0]) ** 2 + (
                self._ys - self.pos[i, 1]
            ) ** 2 <= r * r
            img[amask] = BODY_COLORS[i % len(BODY_COLORS)]
        return img


# --------------------------------------------------------------------- policies
class RepeatRandomPolicy:
    """Random agent actions, each held for `action_repeat` steps (coherent motion)."""

    def __init__(self, cfg: Config = CFG, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._a = 0
        self._left = 0

    def __call__(self, obs=None) -> int:
        if self._left <= 0:
            self._a = int(self.rng.integers(0, self.cfg.n_actions))
            self._left = self.cfg.action_repeat
        self._left -= 1
        return self._a


def rollout(env: MultiBody2DEnv, policy, ep_len: int | None = None):
    """Run one episode.

    Returns (obs[T+1,H,W,3] uint8, acts[T] int64, pos[T+1,N,2], vel[T+1,N,2],
             ke[T+1], goal[2]). Positions/velocities are ground truth, kept for
    latent probes and physics-consistency checks (never shown to the model).
    """
    ep_len = ep_len or env.cfg.ep_len
    o = env.reset()
    obs, acts = [o], []
    pos, vel, ke = [env.pos.copy()], [env.vel.copy()], [env.kinetic_energy()]
    for _ in range(ep_len):
        a = policy(o)
        o, _, done, info = env.step(a)
        obs.append(o); acts.append(a)
        pos.append(env.pos.copy()); vel.append(env.vel.copy()); ke.append(info["ke"])
        if done:
            break
    return (np.stack(obs), np.array(acts, np.int64), np.stack(pos).astype(np.float32),
            np.stack(vel).astype(np.float32), np.array(ke, np.float32), env.goal.copy())


def _save_demo():
    """Render a random rollout to a GIF + montage and report energy conservation."""
    import imageio.v2 as imageio

    cfg = CFG
    cfg.ensure_dirs()
    env = MultiBody2DEnv(cfg, seed=2)
    pol = RepeatRandomPolicy(cfg, seed=2)
    obs, acts, pos, vel, ke, goal = rollout(env, pol, ep_len=120)
    gif = os.path.join(cfg.artifacts_dir, "env2_demo.gif")
    imageio.mimsave(gif, list(obs), fps=20, loop=0)
    sel = obs[::8][:15]
    rows = [np.hstack(list(sel[i : i + 5])) for i in range(0, min(15, len(sel)), 5)]
    montage = np.vstack(rows[: max(1, len(sel) // 5)])
    png = os.path.join(cfg.artifacts_dir, "env2_demo_montage.png")
    imageio.imwrite(png, montage)
    # Energy is only ~conserved when the agent doesn't thrust; report the free-body
    # spread as a sanity check on the collision solver.
    print(f"frames={len(obs)} N={env.N} ke[0]={ke[0]:.4f} ke[-1]={ke[-1]:.4f} "
          f"ke_std={ke.std():.4f}")
    print(f"saved {gif}\nsaved {png}")


if __name__ == "__main__":
    _save_demo()

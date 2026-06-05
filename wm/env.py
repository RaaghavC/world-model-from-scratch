"""A from-scratch 2D physics environment: a thrusted ball in a walled arena.

No gym / gymnasium dependency -- both the dynamics and the renderer are written
here in pure NumPy. The agent is a ball that accelerates under thrust, loses
energy to friction, is speed-capped, and bounces elastically off the four walls.
A static goal square gives the controller something to reach.

Observations are (img_size, img_size, 3) uint8 RGB frames. The domain is
deliberately simple and *learnable*: a competent world model should reproduce the
bounce dynamics and action-conditioned motion almost perfectly, which makes
"dreams" trivial to judge by eye -- the whole point of choosing it.

State (never shown to the model, only used for data labels / reward):
    pos : ball centre in normalized [0, 1]^2 coordinates
    vel : ball velocity in normalized units per step
    goal: goal-square centre in [0, 1]^2 (static within an episode)
"""
from __future__ import annotations

import os

import numpy as np

from wm.config import CFG, Config

# RGB palette (0-255). The background is dark so the bright ball/goal stand out.
C_BG = (18, 18, 28)      # arena background
C_WALL = (70, 70, 95)    # 1px border frame drawn on the outer edge
C_GOAL = (40, 185, 80)   # green goal square
C_AGENT = (245, 130, 45)  # orange agent ball

# Discrete action -> unit acceleration direction (x, y).
# y points DOWN, matching image/row coordinates (row index increases downward).
ACTION_VEC = {
    0: (0.0, 0.0),    # no-op (coast)
    1: (1.0, 0.0),    # thrust +x  (right)
    2: (-1.0, 0.0),   # thrust -x  (left)
    3: (0.0, 1.0),    # thrust +y  (down)
    4: (0.0, -1.0),   # thrust -y  (up)
}


class Particle2DEnv:
    """Deterministic 2D thrust-and-bounce environment.

    The MDP is fully deterministic given the action sequence, which is convenient:
    it means a *deterministic* dream (taking the MDN's most-likely next latent) is
    the right thing to match against reality.
    """

    def __init__(self, cfg: Config = CFG, seed: int | None = None):
        self.cfg = cfg
        self.S = cfg.img_size
        self.rng = np.random.default_rng(seed)
        # Pre-compute, once, the normalized [0,1] coordinate of every pixel centre.
        # These grids are reused every render() call to draw the goal/ball by mask,
        # which is far faster than per-pixel Python loops.
        ys, xs = np.mgrid[0 : self.S, 0 : self.S]
        self._xs = (xs + 0.5) / self.S   # (S,S) x-coordinate of each pixel centre
        self._ys = (ys + 0.5) / self.S   # (S,S) y-coordinate of each pixel centre
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.goal = np.zeros(2)
        self.t = 0
        self.reset()

    # ------------------------------------------------------------------ core MDP
    def reset(self, pos=None, goal=None) -> np.ndarray:
        """Start a new episode. Random ball/goal placement unless pinned via args."""
        r = self.cfg.agent_radius
        # Ball starts anywhere fully inside the walls (>= r from each edge).
        self.pos = (
            self.rng.uniform(r, 1 - r, size=2) if pos is None else np.array(pos, float)
        )
        self.vel = np.zeros(2)
        # Goal kept away from the very edges so it renders fully on-frame.
        self.goal = (
            self.rng.uniform(0.18, 0.82, size=2)
            if goal is None
            else np.array(goal, float)
        )
        self.t = 0
        return self.render()

    def step(self, a: int):
        """Advance one timestep under action `a`. Returns (obs, reward, done, info)."""
        c = self.cfg
        # 1) Apply thrust as an instantaneous acceleration in the action direction.
        ax, ay = ACTION_VEC[int(a)]
        self.vel[0] += ax * c.thrust
        self.vel[1] += ay * c.thrust
        # 2) Friction: retain a fraction of velocity each step. With friction f and
        #    thrust T the terminal speed is ~T/(1-f), giving a bounded, learnable range.
        self.vel *= c.friction
        # 3) Speed cap (keeps motion within the range the model has seen).
        sp = float(np.hypot(*self.vel))
        if sp > c.max_speed:
            self.vel *= c.max_speed / sp
        # 4) Integrate position (explicit Euler).
        self.pos += self.vel
        # 5) Wall collisions: clamp inside [r, 1-r] and reflect velocity with
        #    restitution < 1 so bounces visibly lose energy.
        r = c.agent_radius
        for i in range(2):
            if self.pos[i] < r:
                self.pos[i] = r
                self.vel[i] *= -c.restitution
            elif self.pos[i] > 1 - r:
                self.pos[i] = 1 - r
                self.vel[i] *= -c.restitution
        self.t += 1
        # Reward is dense and shaped as negative distance-to-goal; "reached" fires
        # when the ball centre is within one goal-radius.
        dist = float(np.hypot(*(self.pos - self.goal)))
        reward = -dist
        done = self.t >= c.ep_len
        info = {"dist": dist, "reached": dist < c.goal_radius}
        return self.render(), reward, done, info

    # ------------------------------------------------------------------ rendering
    def render(self) -> np.ndarray:
        """Rasterize the current state to a (S, S, 3) uint8 RGB frame."""
        S = self.S
        img = np.empty((S, S, 3), np.uint8)
        img[:] = C_BG
        # 1px wall border on all four edges.
        img[0, :] = img[-1, :] = C_WALL
        img[:, 0] = img[:, -1] = C_WALL
        # Goal: an axis-aligned square (|dx|<r and |dy|<r) painted by boolean mask.
        gr = self.cfg.goal_radius
        gmask = (np.abs(self._xs - self.goal[0]) < gr) & (
            np.abs(self._ys - self.goal[1]) < gr
        )
        img[gmask] = C_GOAL
        # Agent: a filled disc (dx^2 + dy^2 <= r^2). Drawn last so it sits on top.
        ar = self.cfg.agent_radius
        amask = (self._xs - self.pos[0]) ** 2 + (self._ys - self.pos[1]) ** 2 <= ar * ar
        img[amask] = C_AGENT
        return img


# --------------------------------------------------------------------- policies
class RepeatRandomPolicy:
    """Random actions, each held for `action_repeat` steps.

    Holding an action produces coherent, sustained motion (and wall bounces), so
    the dataset teaches the dynamics model how *directional* thrust moves the ball
    -- exactly the behaviour a goal-seeking controller will later need.
    """

    def __init__(self, cfg: Config = CFG, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._a = 0       # currently-held action
        self._left = 0    # steps remaining on the current action

    def __call__(self, obs=None) -> int:
        if self._left <= 0:
            self._a = int(self.rng.integers(0, self.cfg.n_actions))
            self._left = self.cfg.action_repeat
        self._left -= 1
        return self._a


def rollout(env: Particle2DEnv, policy, ep_len: int | None = None):
    """Run one episode.

    Returns (obs[T+1,H,W,3] uint8, actions[T] int64, pos[T+1,2] float32).
    There is one more observation/position than there are actions (the initial
    frame). Positions are the ground-truth ball coordinates -- used later to train
    a latent->position probe so the controller can be rewarded inside the dream.
    """
    ep_len = ep_len or env.cfg.ep_len
    o = env.reset()
    obs, acts, pos = [o], [], [env.pos.copy()]
    for _ in range(ep_len):
        a = policy(o)
        o, _, done, _ = env.step(a)
        obs.append(o)
        acts.append(a)
        pos.append(env.pos.copy())
        if done:
            break
    return np.stack(obs), np.array(acts, np.int64), np.stack(pos).astype(np.float32)


def _save_demo():
    """Render a random rollout to a GIF and a montage PNG for eyeballing the env."""
    import imageio.v2 as imageio

    cfg = CFG
    cfg.ensure_dirs()
    env = Particle2DEnv(cfg, seed=1)
    pol = RepeatRandomPolicy(cfg, seed=1)
    obs, acts, _ = rollout(env, pol, ep_len=120)
    gif_path = os.path.join(cfg.artifacts_dir, "env_demo.gif")
    imageio.mimsave(gif_path, list(obs), fps=20, loop=0)

    # Montage: every 8th frame, tiled into a 3x5 grid.
    sel = obs[::8][:15]
    rows = [np.hstack(list(sel[i : i + 5])) for i in range(0, 15, 5)]
    montage = np.vstack(rows)
    png_path = os.path.join(cfg.artifacts_dir, "env_demo_montage.png")
    imageio.imwrite(png_path, montage)
    print(f"frames={len(obs)} actions={acts.shape} dist0={np.hypot(*(env.pos-env.goal)):.3f}")
    print(f"saved {gif_path}")
    print(f"saved {png_path}")


if __name__ == "__main__":
    _save_demo()

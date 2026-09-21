"""Population *suitability* screen for the offline-AHT setting -- a pre-flight gate.

Run this on a generated population + its collected dataset **before** training any
offline baseline (BC/LIAM/OMIS/TAO). It answers, cheaply (minutes of logistic
regression, no baseline training), whether the population is even in a regime where
an offline ego can generalize -- catching the two dead ends the expensive suite
would otherwise discover the hard way: adversarial/un-best-respondable populations,
and reactive/non-discriminative ones.

Everything reduces to one environment-agnostic object -- the predictability
structure of the collected trajectories -- estimated by three quantities:

  A  population structure  (deterministic, from the crossplay matrix), two orthogonal
     scalars: **Diversity D** = headroom of per-teammate adaptation over the best fixed
     ego (D->0 trivial, D->1 modelling essential; = ZSC-Eval's BR-Div), and **Sparseness
     S** = how catastrophic a mismatch is (cross/self and dead-fraction). Good testbed =
     high D + low S; high-D+high-S is adversarial, low-D is trivial.

  R(t)  recoverability = I(teammate_id ; history_t): can you tell *who* you're playing
     from t steps of the ego's own local history? A linear probe's held-out accuracy
     over increasing history, vs chance.

  V     value of identity = I(action ; teammate_id | context): does knowing the
     teammate reduce the ego's action uncertainty, beyond what the context already
     gives? Measured at two context levels -- current obs only (reactive) and
     obs + history -- as the cross-entropy drop from adding the teammate id.

The verdict combines them:

  A high                      -> STOP: adversarial minefield; no generalist survives.
                                 Fix the *generator* (lower separation / anchor task reward).
  V(obs)~=V(hist)~=0          -> REACTIVE: BC ~= oracle; env won't discriminate teammate
                                 modelling. Suitable but a control, not a test (spread-like).
  V>0, R stays ~chance        -> INFEASIBLE offline: identity matters but can't be recovered
                                 from history (adversarial/symmetry-broken); needs online
                                 adaptation or symmetry-invariant generation.
  V(obs)~=0, V(hist)>0, R>chance -> HIDDEN-CONVENTION (Hanabi-like): BC will fail, but a
                                 sequence/identity model over history can work. GO with LIAM/TAO.
  V>0, R rises with t         -> GO: identifiable and valuable; teammate modelling should beat BC.

This is a *necessary*-condition screen, not sufficient: a GO does not guarantee a
given baseline finds the policy, but a STOP/INFEASIBLE reliably predicts failure.

Usage:
    uv run python scripts/diagnose_population_suitability.py \
        configs/hanabi/training/pooled_expert_scaled/bc.json \
        [--crossplay populations/hanabi/pooled_crossplay.npz] \
        [--dataset results/.../dataset.vlt] [--max-windows 60000]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

# Verdict thresholds. Heuristic and deliberately conservative -- tune against a few
# known populations. Normalised so they are comparable across environments:
#   *_norm quantities are in [0, 1] (fraction of the achievable headroom).
_ADV_DEAD_FRAC = 0.5  # >half of off-BR responses near their teammate's worst -> minefield
_ADV_RATIO = 0.3  # mean(cross)/mean(self) below this -> heavily specialised
_LOW_DIVERSITY = 0.3  # D below this: best fixed ego captures >70% of the ceiling -> trivial
_R_IDENTIFIABLE = 0.35  # recoverability headroom above chance to call it identifiable
_R_UNIDENTIFIABLE = 0.12  # below this, effectively un-identifiable
_V_MATTERS = 0.03  # accuracy gain from adding id to call identity "valuable"


def _fit_softmax(X, y, n_classes, *, steps=400, seed=0):
    """Held-out accuracy + cross-entropy of a linear softmax probe X -> y.

    A dependency-light stand-in for sklearn's LogisticRegression: full-batch Adam on
    standardised features. Returns (test_accuracy, test_cross_entropy, chance_accuracy).
    Capacity is deliberately low so a *high* score is strong evidence the signal is
    linearly present, not that a big model memorised it.
    """
    import jax
    import jax.numpy as jnp
    import optax

    X = np.asarray(X, np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    y = np.asarray(y, np.int32)
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ntr = max(1, int(0.8 * n))
    tr, te = perm[:ntr], perm[ntr:]
    if len(te) == 0:  # tiny dataset -- fall back to train==test
        te = tr
    Xtr, ytr = jnp.asarray(X[tr]), jnp.asarray(y[tr])
    Xte, yte = jnp.asarray(X[te]), jnp.asarray(y[te])

    params = {"W": jnp.zeros((X.shape[1], n_classes)), "b": jnp.zeros((n_classes,))}
    opt = optax.adam(5e-2)
    opt_state = opt.init(params)

    def ce(p, Xb, yb):
        logits = Xb @ p["W"] + p["b"]
        return optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()

    @jax.jit
    def update(p, s):
        loss, g = jax.value_and_grad(ce)(p, Xtr, ytr)
        u, s = opt.update(g, s)
        return optax.apply_updates(p, u), s, loss

    for _ in range(steps):
        params, opt_state, _ = update(params, opt_state)

    logits = np.asarray(Xte @ params["W"] + params["b"])
    pred = logits.argmax(1)
    acc = float((pred == np.asarray(yte)).mean())
    test_ce = float(np.asarray(ce(params, Xte, yte)))
    # chance = always predict the most frequent training class
    _, counts = np.unique(np.asarray(ytr), return_counts=True)
    chance = float(counts.max() / len(ytr))
    return acc, test_ce, chance


def _bag(obs, act, mask, action_dim):
    """Mask-weighted mean observation + normalised action histogram over the window.

    A permutation-invariant fingerprint of the behaviour in a stretch of history --
    ``obs`` (n, k, d), ``act`` (n, k), ``mask`` (n, k) -> (n, d + action_dim).
    """
    m = mask.astype(np.float32)
    denom = m.sum(1, keepdims=True) + 1e-6
    mean_obs = (obs * m[..., None]).sum(1) / denom
    onehot = (act[..., None] == np.arange(action_dim)).astype(np.float32)
    act_hist = (onehot * m[..., None]).sum(1) / denom
    return np.concatenate([mean_obs, act_hist], -1)


def _load_windows(config_path, max_windows, dataset_override=None):
    from oaht_bench.configs import load_job
    from oaht_bench.dataset.dataset import Dataset

    job = load_job(config_path)
    cfg = job.offline
    ds = Dataset(
        dataset_override or job.dataset_path,
        context_length=cfg.context_length,
        stride=cfg.stride,
        normalize=False,
    )
    w = ds.windows
    n = len(w)
    idx = np.arange(n)
    if n > max_windows:
        idx = np.random.default_rng(0).choice(n, size=max_windows, replace=False)
    obs = np.asarray(w.ego_obs)[idx]  # (N, T, d)
    act = np.asarray(w.ego_actions)[idx].astype(np.int64)  # (N, T)
    mask = np.asarray(w.mask)[idx]  # (N, T)
    tid = np.asarray(w.teammate_id)[idx].astype(np.int64)  # (N,)
    action_dim = int(ds.action_dim)
    return job, obs, act, mask, tid, action_dim


def _crossplay_metrics(npz_path):
    """The two orthogonal, *deterministic* population-quality scalars from the crossplay.

    **Diversity D** = ``1 - best_generalist / oracle_ceiling`` -- the headroom a
    per-teammate-adaptive ego has over the single best FIXED ego. D->0 means one policy
    serves everyone (trivial: modelling can't help); D->1 means each teammate needs a
    different best-response (modelling is essential). (This is ZSC-Eval's BR-Div read off
    the matrix.)

    **Sparseness S** (navigability) = how catastrophic a mismatch is: ``cross/self``
    (low = mismatches collapse coordination) and ``dead_fraction`` (share of off-diagonal
    pairs near the worst response for their teammate -> a minefield).

    A population is a good AHT testbed iff D is HIGH (modelling needed) AND S is LOW
    (mismatches survivable). High-D + high-S is the un-generalizable adversarial corner;
    low-D is the trivial/non-discriminative corner.
    """
    from oaht_bench.population.pooled_crossplay import normalise_per_teammate

    if str(npz_path).endswith(".csv"):
        import pandas as pd

        m = pd.read_csv(npz_path, index_col=0).values.astype(float)
    else:
        m = np.asarray(np.load(npz_path, allow_pickle=True)["matrix"], float)
    k = m.shape[0]
    diag = np.diag(m)
    off = m[~np.eye(k, dtype=bool)]
    ratio = float(off.mean() / diag.mean()) if diag.mean() != 0 else float("nan")
    # Column-normalised: for each teammate, best ego -> 1, worst -> 0. Off-diagonal
    # cells that map near 0 are "dead" responses (near-worst for their teammate).
    q = normalise_per_teammate(m)
    off_q = q[~np.eye(k, dtype=bool)]
    dead_frac = float((off_q < 0.1).mean())
    best_generalist = float(m.mean(1).max())  # best single FIXED ego over all teammates
    oracle_ceiling = float(m.max(0).mean())  # mean over teammates of the best ego for THAT teammate
    diversity = 1 - best_generalist / oracle_ceiling if oracle_ceiling != 0 else float("nan")
    return {
        "roster_size": k,
        "self_mean": float(diag.mean()),
        "cross_mean": float(off.mean()),
        "diversity": float(diversity),
        "best_generalist": best_generalist,
        "oracle_ceiling": oracle_ceiling,
        "cross_over_self": ratio,
        "dead_fraction": dead_frac,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "config", help="an offline training job JSON (for its dataset + context_length)"
    )
    ap.add_argument(
        "--crossplay",
        default=None,
        help="pooled_crossplay.npz (default: infer from populations/<env>/)",
    )
    ap.add_argument("--max-windows", type=int, default=60000, help="subsample cap for the probes")
    ap.add_argument("--probe-steps", type=int, default=400)
    ap.add_argument("--dataset", default=None, help="override the config's dataset_path (a .vlt) -- "
                    "run on any collection without editing a config; context_length/stride still come from the config")
    args = ap.parse_args()

    job, obs, act, mask, tid, action_dim = _load_windows(args.config, args.max_windows, args.dataset)
    env_name = job.env.name
    T = obs.shape[1]

    # Dense, contiguous teammate labels for the classifier.
    uniq = np.unique(tid)
    remap = {int(t): i for i, t in enumerate(uniq.tolist())}
    y = np.array([remap[int(t)] for t in tid], np.int64)
    n_teammates = len(uniq)
    print(f"\n=== population suitability: {env_name} ===")
    print(f"windows={len(y)}  context_length={T}  teammates={n_teammates}  action_dim={action_dim}")

    # Keep windows whose current (last) step is real; history excludes that step so
    # the action target never leaks into its own features.
    cur = mask[:, -1].astype(bool)
    cur_obs = obs[cur, -1]
    target = act[cur, -1]
    y_cur = y[cur]
    hist_obs, hist_act, hist_mask = obs[cur, :-1], act[cur, :-1], mask[cur, :-1]

    # --- R(t): recoverability of teammate identity from t steps of history ---
    print("\n[R] recoverability  I(teammate_id ; history_t)   (chance-normalised)")
    r_curve = {}
    if n_teammates < 2:
        print("  only one teammate present -- recoverability undefined (need the pooled dataset).")
    else:
        for frac in (0.25, 0.5, 1.0):
            k = max(1, int(round(frac * hist_obs.shape[1])))
            feat = _bag(hist_obs[:, -k:], hist_act[:, -k:], hist_mask[:, -k:], action_dim)
            acc, _, chance = _fit_softmax(feat, y_cur, n_teammates, steps=args.probe_steps)
            norm = (acc - chance) / (1 - chance + 1e-9)
            r_curve[frac] = norm
            print(
                f"  t={frac:>4.0%} history:  acc={acc:.3f}  (chance {chance:.3f})  headroom R_norm={norm:.3f}"
            )
    r_full = r_curve.get(1.0, 0.0)

    # --- V: value of teammate identity for predicting the ego's next action ---
    print("\n[V] value of identity  I(action ; teammate_id | context)   (accuracy gain from id)")
    id_oh = (y_cur[:, None] == np.arange(n_teammates)).astype(np.float32)
    hist_feat = _bag(hist_obs, hist_act, hist_mask, action_dim)

    def _gain(base_feat):
        a0, ce0, _ = _fit_softmax(base_feat, target, action_dim, steps=args.probe_steps)
        a1, ce1, _ = _fit_softmax(
            np.concatenate([base_feat, id_oh], 1), target, action_dim, steps=args.probe_steps
        )
        return a1 - a0, ce0 - ce1

    v_obs_gain, v_obs_ce = _gain(cur_obs)
    v_hist_gain, v_hist_ce = _gain(np.concatenate([cur_obs, hist_feat], 1))
    print(f"  given current obs:      acc gain={v_obs_gain:+.3f}  CE drop={v_obs_ce:+.3f}")
    print(f"  given obs + history:    acc gain={v_hist_gain:+.3f}  CE drop={v_hist_ce:+.3f}")

    # --- A: population structure (diversity D + sparseness S) from the crossplay ---
    print("\n[A] population structure  (crossplay -- deterministic)")
    xpath = args.crossplay or f"populations/{env_name}/pooled_crossplay.npz"
    adv = None
    try:
        adv = _crossplay_metrics(xpath)
        print(
            f"  roster={adv['roster_size']}  Diversity D={adv['diversity']:.2f}  "
            f"(best-generalist {adv['best_generalist']:.3f} / oracle-ceiling {adv['oracle_ceiling']:.3f})"
        )
        print(
            f"  Sparseness S: cross/self={adv['cross_over_self']:.2f}  "
            f"dead-fraction={adv['dead_fraction']:.2f}  (high = mismatches are a minefield)"
        )
    except FileNotFoundError:
        print(f"  crossplay matrix not found at {xpath} -- skipping (pass --crossplay).")

    # --- verdict ---
    adversarial = adv is not None and (
        adv["dead_fraction"] > _ADV_DEAD_FRAC or adv["cross_over_self"] < _ADV_RATIO
    )
    low_diversity = adv is not None and adv["diversity"] < _LOW_DIVERSITY
    id_valuable = max(v_obs_gain, v_hist_gain) > _V_MATTERS
    hidden_convention = v_obs_gain <= _V_MATTERS < v_hist_gain
    identifiable = r_full > _R_IDENTIFIABLE
    unidentifiable = r_full < _R_UNIDENTIFIABLE

    print("\n=== VERDICT ===")
    if adversarial:
        v = (
            "STOP -- adversarial minefield. Cross-play collapses toward the worst response, "
            "so no single offline ego can generalize. Fix the GENERATOR: lower cross_play_weight / "
            "raise the task-reward anchor, or select a representative (BR-Div) subset."
        )
    elif low_diversity or not id_valuable:
        reasons = []
        if low_diversity:
            reasons.append(f"population diversity is low (D={adv['diversity']:.2f} -- one fixed ego serves all)")
        if not id_valuable:
            reasons.append("teammate identity barely changes the ego's action")
        v = (
            "REACTIVE / non-discriminative -- "
            + "; ".join(reasons)
            + ". BC ~= the teammate-id oracle, so teammate-modelling can't help. A useful CONTROL, "
            "not a discriminative tier (spread-like)."
        )
    elif unidentifiable:
        v = (
            "INFEASIBLE offline. Identity is VALUABLE but not RECOVERABLE from history -- an adversarial / "
            "symmetry-broken convention. No offline method can switch on a signal that isn't there; needs "
            "online adaptation or symmetry-invariant generation."
        )
    elif hidden_convention and not unidentifiable:
        v = (
            "HIDDEN-CONVENTION (Hanabi-like). Identity is useless from the current obs but valuable given "
            "history, and is recoverable -> BC will fail, but a sequence/identity model (LIAM/TAO) can work. "
            "GO, but expect BC to be the wrong baseline."
        )
    elif identifiable and id_valuable:
        v = (
            "GO. Conventions are identifiable from history AND identity is valuable -> teammate-modelling "
            "should beat BC. This is the target regime for the offline suite."
        )
    else:
        v = (
            "MARGINAL. Identity has some value but recoverability is weak; expect a modest teammate-modelling "
            "edge over BC and a large generalization gap. Inspect R(t) and consider a lower-separation regen."
        )
    print(v)
    print(
        "\n(Necessary-condition screen: a GO does not guarantee any single baseline succeeds, "
        "but STOP/INFEASIBLE reliably predicts failure. Thresholds are heuristic -- see the source.)"
    )

    # machine-readable summary for logging / sweeps
    print(
        "\nSUMMARY "
        + json.dumps(
            {
                "env": env_name,
                "teammates": n_teammates,
                "diversity_D": round(adv["diversity"], 4) if adv else None,
                "sparseness": {"cross_over_self": round(adv["cross_over_self"], 4),
                               "dead_fraction": round(adv["dead_fraction"], 4)} if adv else None,
                "R_norm": {str(k): round(v, 4) for k, v in r_curve.items()},
                "V_obs_gain": round(v_obs_gain, 4),
                "V_hist_gain": round(v_hist_gain, 4),
                "crossplay": adv,
                "verdict": v.split(".")[0].split("--")[0].strip(),
            }
        )
    )


if __name__ == "__main__":
    main()

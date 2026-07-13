"""
Scientific comparison of real vs. simulated animal movement (velocity) distributions.

This pipeline answers two questions rigorously:
  1. Are the real (KABR) and simulated velocity distributions statistically different,
     and by how much (effect sizes / distances, not just p-values)?
  2. Which simulation parameters (intrinsicForwardCoefficient, intrinsicLateralMultiplier)
     bring the simulated distribution closest to the real one?

Key methodological choices (documented so the analysis is reproducible / defensible):

  * UNITS. Both datasets are divided by 3.6. The real reconstruction is stored in km/h;
    we assume the simulation is stored in the same unit. Because BOTH are scaled by the
    same factor, the *shape* comparison and all distance metrics are invariant to this
    choice; only the absolute m/s labels depend on it.

  * INITIALISATION ARTEFACT. At t=0 every simulated agent has velocity exactly 0
    (10% of the raw simulated samples). This is a model initial condition, not a
    behaviour, and it injects a spurious spike at 0. We DROP t=0 from the simulation.

  * INSTRUMENT CENSORING. The real velocity is floored at 1 km/h (its minimum is exactly
    1/3.6 m/s): the tracker cannot resolve slower motion. To compare like with like we
    apply the SAME censoring to the simulation (drop samples below the floor). Metrics
    are reported on this "observable" support. Raw simulated stats are also printed so
    the censored mass is visible.

  * SAMPLE-SIZE IMBALANCE. Real N is ~1e2, simulated N is ~1e3-1e5. Classical p-values
    (KS/AD) are driven to ~0 by N alone and are therefore reported but NOT used as the
    primary evidence. The primary metric is the Wasserstein (earth-mover) distance in
    m/s, which is interpretable and scale-honest, complemented by energy distance and
    Jensen-Shannon divergence.
"""

import os
import re
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon

# ----------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------
KMH_TO_MS = 1.0 / 3.6
FLOOR_KMH = 1.0                      # real-instrument velocity floor
FLOOR_MS = FLOOR_KMH * KMH_TO_MS     # ~0.2778 m/s
DROP_INIT_TIME = True                # drop simulation t=0 initialisation artefact
MATCH_CENSORING = True               # apply the real 1 km/h floor to the simulation too
N_BOOT = 2000                        # bootstrap resamples for CIs
RNG = np.random.default_rng(42)
KDE_BW = 1.8                         # KDE smoothing for the comparison figures (same as curve_similarity)

# All comparison figures + the report land in their own folder so they are easy to
# inspect on their own and never mix with the process.py charts in "charts/".
# Override OUT_DIR / PICKLE_PREFIX to analyse a second dataset (e.g. the realistic
# variant) without clobbering the first run's outputs.
OUT_DIR = os.environ.get("OUT_DIR", "analysis")
PICKLE_PREFIX = os.environ.get("PICKLE_PREFIX", "data_summary")

# Viridis-aligned palette. The simulated representatives are ORDERED (low/best/high
# forward coefficient), so sampling them along viridis is a legitimate sequential use;
# the real reference is black ink for maximum separation from the model curves.
# CVD separation of the sim triple passes (ΔE ~33); the light-yellow end is avoided so
# the lines stay readable on a white surface, and every panel carries a legend.
SEQ_CMAP = "viridis"                          # sequential ramp (magnitude) for heatmaps/hue
C_REAL = "#000000"                            # real reference: black ink
C_SIM = ["#46337e", "#25848e", "#3fbc73"]     # viridis @ ~0.15 / 0.45 / 0.72 (low/best/high)
C_BEST = "#25848e"                            # the best-fit simulated curve (viridis teal)
C_HL = "#de3f82"                              # magenta highlight for the best-fit marker
C_OVL = "#addc90"                             # soft viridis-green fill for the overlap area


# Shared paper style — identical to the one used in process.py so every figure across
# both scripts looks like it belongs to the same paper (same theme, fonts, weights).
def apply_paper_style():
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType (journal-safe), avoid Type-3
        "savefig.bbox": "tight",
        "axes.titlesize": "medium",
        "legend.frameon": True,
    })


# ----------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------
def load_real():
    """Real KABR velocity reconstruction, long format, in m/s.

    The reconstruction is floored at 1 km/h: ~20% of samples sit exactly on that floor,
    which is the tracker's minimum resolvable speed (animals "stationary" / below
    resolution), not a measured speed. We drop that spike so the reference is the
    distribution of genuinely moving animals — the same treatment applied to the sim.
    """
    df = pd.read_csv(
        "data/velocity_reconstruction.csv", delimiter=" ", comment="#",
        names=list(range(10)),
    )
    df.rename(columns={0: "time"}, inplace=True)
    df = df.melt(id_vars=["time"], var_name="node", value_name="velocity").dropna()
    df["velocity"] = df["velocity"] * KMH_TO_MS
    if MATCH_CENSORING:
        df = df[df["velocity"] > FLOOR_MS].copy()
    df["source"] = "Real (KABR)"
    return df


def load_sim():
    """Simulated velocities, long format, in m/s. Drops the t=0 initialisation spike."""
    means = pickle.load(open(f"{PICKLE_PREFIX}_mean", "rb"))
    v = means["velocity_simulation"]
    df = v.to_dataframe().reset_index()
    # Keep ONLY scalar velocity-magnitude columns ("node-<int>"). The dataset also
    # contains position components ("node-<int>[x]", "node-<int>[y]") on a different
    # scale (hundreds, signed) that must NOT be pooled with velocities.
    node_cols = [c for c in df.columns if re.fullmatch(r"node-\d+", c)]
    idv = ["time", "intrinsicForwardCoefficient", "intrinsicLateralMultiplier", "NumberOfHerds"]
    df = df.melt(id_vars=idv, value_vars=node_cols, var_name="node", value_name="velocity").dropna()
    df["velocity"] = df["velocity"] * KMH_TO_MS
    if DROP_INIT_TIME:
        df = df[df["time"] > 0].copy()
    return df


def censor(v):
    """Apply the real instrument floor to an array of velocities (m/s).

    Drops the mass at/below the 1 km/h floor so the simulated support matches the real
    one after its floor spike has been removed (strictly-greater keeps the two aligned).
    """
    v = np.asarray(v, dtype=float)
    return v[v > FLOOR_MS] if MATCH_CENSORING else v


# ----------------------------------------------------------------------------------
# Distribution distance / test battery
# ----------------------------------------------------------------------------------
def js_divergence(a, b, bins):
    """Jensen-Shannon divergence (bits) between two samples on a shared bin grid."""
    pa, _ = np.histogram(a, bins=bins, density=False)
    pb, _ = np.histogram(b, bins=bins, density=False)
    pa = pa + 1e-9
    pb = pb + 1e-9
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    return jensenshannon(pa, pb, base=2) ** 2  # squared JS = divergence


def curve_similarity(real, sim, n_grid=512, bw="scott"):
    """Similarity measured on the *density curves themselves* (shared-grid KDEs).

    Complements the sample-based distances in `compare`: these operate on the smooth
    estimated densities f_real, f_sim on a common grid, which is exactly what the eye
    compares when looking at two overlaid KDE curves.

    Returns
    -------
    dict with:
      * overlap       — overlapping coefficient OVL = ∫ min(f,g); 1 = identical curves,
                        0 = disjoint. The most intuitive "how much do the curves coincide".
      * bhattacharyya — Bhattacharyya distance -ln ∫ √(fg); 0 = identical.
      * hellinger     — Hellinger distance √(1 - ∫√(fg)) in [0,1]; 0 = identical.
      * tv            — total-variation distance ½∫|f-g| in [0,1]; 0 = identical.
      * pearson       — Pearson correlation between the two density curves (shape match).
      * cosine        — cosine similarity between the two density vectors.
      * jeffreys      — symmetric KL (Jeffreys) divergence; 0 = identical.
    """
    real = np.asarray(real, float)
    sim = np.asarray(sim, float)
    lo = min(real.min(), sim.min())
    hi = max(real.max(), sim.max())
    grid = np.linspace(lo, hi, n_grid)
    fr = stats.gaussian_kde(real, bw_method=bw)(grid)
    fs = stats.gaussian_kde(sim, bw_method=bw)(grid)
    dx = grid[1] - grid[0]
    # Normalise both to proper densities on the (uniform) grid: ∫ f dx = 1.
    fr = fr / (fr.sum() * dx)
    fs = fs / (fs.sum() * dx)
    pr, ps = fr * dx, fs * dx  # discrete probability masses per cell

    ovl = float(np.sum(np.minimum(pr, ps)))
    bc = float(np.sum(np.sqrt(pr * ps)))                       # Bhattacharyya coefficient
    bhatt = -np.log(bc) if bc > 0 else np.inf
    hell = float(np.sqrt(max(0.0, 1.0 - bc)))
    tv = float(0.5 * np.sum(np.abs(pr - ps)))
    pear = float(np.corrcoef(fr, fs)[0, 1])
    cos = float(np.dot(fr, fs) / (np.linalg.norm(fr) * np.linalg.norm(fs)))
    eps = 1e-12
    prr = (pr + eps) / (pr + eps).sum()
    pss = (ps + eps) / (ps + eps).sum()
    jeffreys = float(np.sum(prr * np.log(prr / pss)) + np.sum(pss * np.log(pss / prr)))
    return {
        "overlap": ovl, "bhattacharyya": float(bhatt), "hellinger": hell,
        "tv": tv, "pearson": pear, "cosine": cos, "jeffreys": jeffreys,
    }


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan


def compare(real, sim):
    """Full test/effect-size battery between one real and one simulated sample (m/s)."""
    real = np.asarray(real, float)
    sim = np.asarray(sim, float)
    lo = min(real.min(), sim.min())
    hi = max(real.max(), sim.max())
    bins = np.linspace(lo, hi, 41)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ks = stats.ks_2samp(real, sim)
        try:
            ad = stats.anderson_ksamp([real, sim])
            ad_stat, ad_p = ad.statistic, ad.significance_level
        except Exception:
            ad_stat, ad_p = np.nan, np.nan
        mw = stats.mannwhitneyu(real, sim, alternative="two-sided")
        cvm = stats.cramervonmises_2samp(real, sim)

    cs = curve_similarity(real, sim)

    return {
        "n_sim": len(sim),
        # --- primary: interpretable distances ---
        "wasserstein": stats.wasserstein_distance(real, sim),      # m/s
        "energy": stats.energy_distance(real, sim),
        "js_div": js_divergence(real, sim, bins),
        # --- curve-shape similarity (on the density curves themselves) ---
        "overlap": cs["overlap"], "bhattacharyya": cs["bhattacharyya"],
        "hellinger": cs["hellinger"], "tv": cs["tv"],
        "pearson": cs["pearson"], "cosine": cs["cosine"], "jeffreys": cs["jeffreys"],
        # --- moments ---
        "mean_sim": np.mean(sim), "std_sim": np.std(sim, ddof=1),
        "skew_sim": stats.skew(sim), "kurt_sim": stats.kurtosis(sim),
        "d_mean": np.mean(sim) - np.mean(real),
        "d_std": np.std(sim, ddof=1) - np.std(real, ddof=1),
        "cohens_d": cohens_d(sim, real),
        # --- classical tests (p-values inflated by N; reported for completeness) ---
        "ks_stat": ks.statistic, "ks_p": ks.pvalue,
        "ad_stat": ad_stat, "ad_p": ad_p,
        "mw_p": mw.pvalue,
        "cvm_stat": cvm.statistic, "cvm_p": cvm.pvalue,
    }


def bootstrap_wasserstein(real, sim, n_boot=N_BOOT):
    """Bootstrap CI for the Wasserstein distance (resampling both samples)."""
    real = np.asarray(real, float)
    sim = np.asarray(sim, float)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        rb = RNG.choice(real, size=len(real), replace=True)
        sb = RNG.choice(sim, size=len(sim), replace=True)
        vals[i] = stats.wasserstein_distance(rb, sb)
    return np.percentile(vals, [2.5, 50, 97.5])


# ----------------------------------------------------------------------------------
# Parameter sweep: distance to real over the full (forward, lateral) grid
# ----------------------------------------------------------------------------------
def parameter_sweep(real_v, sim_df):
    """Distance + curve-similarity to real for every (forward, lateral) combination.

    Computes, per parameter cell: Wasserstein & KS distance (sample-based) and the
    curve-overlap (OVL) & Pearson shape-correlation (density-curve-based), so the whole
    grid can be ranked by any of them and the best configurations tabulated.
    """
    rows = []
    for (fwd, lat), g in sim_df.groupby(["intrinsicForwardCoefficient", "intrinsicLateralMultiplier"]):
        sv = censor(g["velocity"].values)
        if len(sv) < 20:
            continue
        cs = curve_similarity(real_v, sv)
        rows.append({
            "forward": fwd, "lateral": lat,
            "wasserstein": stats.wasserstein_distance(real_v, sv),
            "ks": stats.ks_2samp(real_v, sv).statistic,
            "overlap": cs["overlap"],
            "pearson": cs["pearson"],
            "hellinger": cs["hellinger"],
            "mean_sim": np.mean(sv),
            "std_sim": np.std(sv, ddof=1),
            "median_sim": np.median(sv),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------
def _smooth_density(x, grid, smooth=1.8):
    """Normalised gaussian-KDE density of x on `grid`, deliberately over-smoothed."""
    kde = stats.gaussian_kde(x, bw_method="scott")
    kde.set_bandwidth(kde.factor * smooth)
    d = kde(grid)
    dx = grid[1] - grid[0]
    return d / (d.sum() * dx)


def figure_distributions(real_df, sim_reps, labels):
    """KDE + ECDF + QQ + violin for representative parameter sets (no titles, paper-ready)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    real_v = real_df["velocity"].values
    xmax = np.percentile(np.concatenate([real_v] + [s for s in sim_reps]), 99.5)

    def tag(ax, t):
        ax.text(0.015, 0.97, t, transform=ax.transAxes, ha="left", va="top",
                fontsize=15, fontweight="bold")

    # (a) KDE
    ax = axes[0, 0]
    sns.kdeplot(x=real_v, ax=ax, color=C_REAL, lw=3, fill=True, alpha=0.12,
                bw_adjust=KDE_BW, label="Real (KABR)")
    for s, lab, c in zip(sim_reps, labels, C_SIM):
        sns.kdeplot(x=s, ax=ax, color=c, lw=2.5, bw_adjust=KDE_BW, label=lab)
    ax.set_xlabel("Velocity (m/s)"); ax.set_ylabel("Density")
    ax.set_xlim(0, xmax); ax.legend(fontsize=10, frameon=True, framealpha=0.9)
    tag(ax, "(a)")

    # (b) ECDF
    ax = axes[0, 1]
    sns.ecdfplot(x=real_v, ax=ax, color=C_REAL, lw=3, label="Real (KABR)")
    for s, lab, c in zip(sim_reps, labels, C_SIM):
        sns.ecdfplot(x=s, ax=ax, color=c, lw=2.5, label=lab)
    ax.set_xlabel("Velocity (m/s)"); ax.set_ylabel("Cumulative probability")
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1.02); ax.legend(fontsize=10)
    tag(ax, "(b)")

    # (c) QQ plot (sim quantiles vs real quantiles)
    ax = axes[1, 0]
    q = np.linspace(0.01, 0.99, 99)
    rq = np.quantile(real_v, q)
    lim = 0
    for s, lab, c in zip(sim_reps, labels, C_SIM):
        sq = np.quantile(s, q)
        ax.plot(rq, sq, "o", color=c, ms=5, alpha=0.8, label=lab)
        lim = max(lim, rq.max(), sq.max())
    ax.plot([0, lim], [0, lim], "--", color="#7a7a7a", lw=1.5, label="y = x (identical)")
    ax.set_xlabel("Real quantiles (m/s)"); ax.set_ylabel("Simulated quantiles (m/s)")
    ax.legend(fontsize=10)
    tag(ax, "(c)")

    # (d) Violin
    ax = axes[1, 1]
    data = [real_v] + list(sim_reps)
    names = ["Real (KABR)"] + labels
    parts = ax.violinplot(data, vert=False, showmedians=True, showextrema=False)
    for pc, c in zip(parts["bodies"], [C_REAL] + C_SIM):
        pc.set_facecolor(c); pc.set_alpha(0.55); pc.set_edgecolor(c)
    parts["cmedians"].set_color("#0b0b0b")
    ax.set_yticks(range(1, len(names) + 1)); ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Velocity (m/s)"); ax.set_xlim(0, xmax)
    tag(ax, "(d)")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "velocity_distribution_comparison.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def figure_sweep(sweep_df, real_v, best):
    """Overlap heatmap + overlap-vs-forward curves + moment matching (no titles)."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    real_mean = real_v.mean()
    real_std = real_v.std(ddof=1)
    cmap = plt.get_cmap(SEQ_CMAP)

    def tag(ax, t):
        ax.text(0.02, 1.03, t, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=15, fontweight="bold")

    # (a) Heatmap of curve overlap (OVL, higher = better) over (forward x lateral)
    ax = axes[0]
    piv = sweep_df.pivot(index="lateral", columns="forward", values="overlap")
    im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap=SEQ_CMAP)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"{v:.2f}" for v in piv.index])
    xt = np.linspace(0, len(piv.columns) - 1, 8).astype(int)
    ax.set_xticks(xt); ax.set_xticklabels([f"{piv.columns[i]:.2f}" for i in xt], rotation=45)
    bi = list(piv.index).index(best["lateral"])
    bj = list(piv.columns).index(best["forward"])
    ax.plot(bj, bi, "*", color=C_HL, ms=24, markeredgecolor="white", markeredgewidth=1.4)
    ax.set_xlabel("intrinsicForwardCoefficient"); ax.set_ylabel("intrinsicLateralMultiplier")
    fig.colorbar(im, ax=ax, label="Curve overlap (OVL)")
    tag(ax, "(a)")

    # (b) Overlap vs forward coefficient, one viridis line per lateral level
    ax = axes[1]
    laterals = sorted(sweep_df["lateral"].unique())
    for i, lat in enumerate(laterals):
        c = cmap(i / max(1, len(laterals) - 1))
        g = sweep_df[sweep_df["lateral"] == lat].sort_values("forward")
        ax.plot(g["forward"], g["overlap"], "-o", color=c, ms=4, lw=2, label=f"lateral={lat:.2f}")
    ax.axvline(best["forward"], color=C_HL, ls="--", lw=1.5, alpha=0.8)
    ax.set_xlabel("intrinsicForwardCoefficient"); ax.set_ylabel("Curve overlap (OVL)")
    ax.legend(fontsize=10)
    tag(ax, "(b)")

    # (c) Moment matching: sim mean/SD vs forward, with real reference lines
    ax = axes[2]
    g = sweep_df[sweep_df["lateral"] == best["lateral"]].sort_values("forward")
    ax.plot(g["forward"], g["mean_sim"], "-o", color=C_SIM[1], ms=4, lw=2, label="Sim mean")
    ax.plot(g["forward"], g["std_sim"], "-o", color=C_SIM[2], ms=4, lw=2, label="Sim SD")
    ax.axhline(real_mean, color=C_REAL, ls="--", lw=2, label=f"Real mean ({real_mean:.3f})")
    ax.axhline(real_std, color="#7a7a7a", ls=":", lw=2, label=f"Real SD ({real_std:.3f})")
    ax.axvline(best["forward"], color=C_HL, ls="--", lw=1.5, alpha=0.6)
    ax.set_xlabel(f"intrinsicForwardCoefficient (lateral={best['lateral']:.2f})")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend(fontsize=10)
    tag(ax, "(c)")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "parameter_sweep.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def figure_overlap(real_v, sv_best, best, cs):
    """Curve similarity: the two smoothed density curves with their overlap shaded.

    The shaded area IS the overlapping coefficient (OVL). Deliberately over-smoothed
    so the comparison reads cleanly; a compact box reports the key similarity metrics.
    """
    lo = min(real_v.min(), sv_best.min())
    # Extend the x-axis to the far right tail so both full supports (and the simulated
    # fast tail past the real max) are visible, not just the bulk.
    hi = max(real_v.max(), np.percentile(np.concatenate([real_v, sv_best]), 99.9))
    grid = np.linspace(lo, hi, 512)
    fr = _smooth_density(real_v, grid)
    fs = _smooth_density(sv_best, grid)
    fmin = np.minimum(fr, fs)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    # No legend entry for the shading — the overlap is shown visually, not labelled.
    ax.fill_between(grid, fmin, color=C_OVL, alpha=0.55, lw=0)
    ax.plot(grid, fr, color=C_REAL, lw=3.5, label="Real (KABR)")
    ax.plot(grid, fs, color=C_BEST, lw=3.5, label="Simulation (best)")
    ax.set_xlim(0, hi); ax.set_ylim(bottom=0)
    ax.set_xlabel("Velocity (m/s)"); ax.set_ylabel("Density")
    ax.text(0.97, 0.95, f"Pearson  {cs['pearson']:.3f}", transform=ax.transAxes,
            ha="right", va="top", family="monospace", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.95))
    ax.legend(loc="center right", fontsize=12, frameon=True, framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "curve_similarity.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    apply_paper_style()
    print("Loading data...")
    real_df = load_real()
    sim_df = load_sim()
    real_v = real_df["velocity"].values

    # Raw (uncensored) simulated stats, to expose the mass hidden by censoring.
    raw_sim = sim_df["velocity"].values
    print(f"\nReal N={len(real_v)} | Simulated N(raw, t>0)={len(raw_sim)}")
    print(f"Fraction of raw sim below the {FLOOR_KMH} km/h floor: "
          f"{np.mean(raw_sim < FLOOR_MS):.1%}")

    # -----------------------------------------------------------------
    # 1. Full parameter sweep -> find best-fitting parameters
    # -----------------------------------------------------------------
    print("\nRunning parameter sweep over the full grid...")
    sweep = parameter_sweep(real_v, sim_df)
    sweep = sweep.sort_values("wasserstein").reset_index(drop=True)
    best = sweep.iloc[0].to_dict()
    print(f"Best fit: forward={best['forward']:.3f}, lateral={best['lateral']:.2f} "
          f"-> Wasserstein={best['wasserstein']:.4f} m/s")

    # -----------------------------------------------------------------
    # 2. Representative comparison (best fit + a low and a high forward)
    # -----------------------------------------------------------------
    lat0 = best["lateral"]
    fwds = sorted(sim_df["intrinsicForwardCoefficient"].unique())
    rep_fwds = [fwds[0], best["forward"], fwds[-1]]
    rep_fwds = sorted(set(rep_fwds))
    sim_reps, labels, stat_rows = [], [], []
    for fwd in rep_fwds:
        g = sim_df[(sim_df["intrinsicForwardCoefficient"] == fwd) &
                   (sim_df["intrinsicLateralMultiplier"] == lat0)]
        sv = censor(g["velocity"].values)
        if len(sv) < 20:
            print(f"  (skipping forward={fwd:.3f}: only {len(sv)} samples above the "
                  f"{FLOOR_KMH} km/h floor — animals too slow to be observable)")
            continue
        sim_reps.append(sv)
        tag = "best" if np.isclose(fwd, best["forward"]) else ""
        labels.append(f"Sim fwd={fwd:.2f}{' ★' if tag else ''}")
        res = compare(real_v, sv)
        res["forward"] = fwd; res["lateral"] = lat0
        stat_rows.append(res)

    # -----------------------------------------------------------------
    # 3. Bootstrap CI for the best fit
    # -----------------------------------------------------------------
    gbest = sim_df[(np.isclose(sim_df["intrinsicForwardCoefficient"], best["forward"])) &
                   (sim_df["intrinsicLateralMultiplier"] == lat0)]
    sv_best = censor(gbest["velocity"].values)
    ci = bootstrap_wasserstein(real_v, sv_best)
    print(f"Bootstrap Wasserstein 95% CI at best fit: "
          f"[{ci[0]:.4f}, {ci[2]:.4f}] (median {ci[1]:.4f}) m/s")

    # -----------------------------------------------------------------
    # 4. Figures
    # -----------------------------------------------------------------
    print("\nRendering figures...")
    cs_best = curve_similarity(real_v, sv_best)
    print(f"Curve overlap (OVL) at best fit: {cs_best['overlap']:.3f} "
          f"(Hellinger {cs_best['hellinger']:.3f}, TV {cs_best['tv']:.3f}, "
          f"Pearson {cs_best['pearson']:.3f})")
    f1 = figure_distributions(real_df, sim_reps, labels)
    f2 = figure_sweep(sweep, real_v, best)
    f3 = figure_overlap(real_v, sv_best, best, cs_best)
    print(f"  {f1}\n  {f2}\n  {f3}")

    # -----------------------------------------------------------------
    # 5. Report
    # -----------------------------------------------------------------
    write_report(real_v, raw_sim, sweep, best, ci, stat_rows, labels, cs_best)
    print(f"\nReport written to {OUT_DIR}/analysis_report.md")
    write_latex_tables(real_v, sweep, best, stat_rows, labels)
    print(f"LaTeX tables written to {OUT_DIR}/tables.tex")


def _interpret(js):
    if js < 0.02:
        return "near-identical"
    if js < 0.1:
        return "similar"
    if js < 0.3:
        return "moderately different"
    return "strongly different"


def write_report(real_v, raw_sim, sweep, best, ci, stat_rows, labels, cs_best):
    real_mean, real_std = real_v.mean(), real_v.std(ddof=1)
    real_med = np.median(real_v)
    best_row = next(r for r in stat_rows if np.isclose(r["forward"], best["forward"]))

    lines = []
    lines.append("# Real vs. Simulated Velocity Distributions — Analysis Report\n")
    lines.append("## Method\n")
    lines.append(f"- Units: velocity in m/s (raw km/h ÷ 3.6).")
    lines.append(f"- Simulation t=0 initialisation spike **removed**.")
    lines.append(f"- Real instrument censored at {FLOOR_KMH} km/h ({FLOOR_MS:.3f} m/s); "
                 f"the same floor is applied to the simulation for a fair comparison "
                 f"(**{np.mean(raw_sim < FLOOR_MS):.1%}** of raw simulated samples fall below it).")
    lines.append(f"- Real N = {len(real_v)}. Primary metric: Wasserstein (earth-mover) "
                 f"distance in m/s; classical p-values are reported but are inflated by the "
                 f"large simulated N and are not the primary evidence.\n")

    lines.append("## Real distribution (reference)\n")
    lines.append(f"| mean | SD | median | min | max |")
    lines.append(f"|---|---|---|---|---|")
    lines.append(f"| {real_mean:.3f} | {real_std:.3f} | {real_med:.3f} | "
                 f"{real_v.min():.3f} | {real_v.max():.3f} |\n")

    lines.append("## Best-fitting parameters\n")
    lines.append(f"**intrinsicForwardCoefficient = {best['forward']:.3f}**, "
                 f"**intrinsicLateralMultiplier = {best['lateral']:.2f}**\n")
    lines.append(f"- Wasserstein distance to real: **{best['wasserstein']:.4f} m/s** "
                 f"(bootstrap 95% CI [{ci[0]:.4f}, {ci[2]:.4f}]).")
    lines.append(f"- At this setting: sim mean {best['mean_sim']:.3f} m/s "
                 f"(real {real_mean:.3f}), sim SD {best['std_sim']:.3f} (real {real_std:.3f}).")
    lines.append(f"- Distribution similarity (JS divergence {best_row['js_div']:.3f}): "
                 f"**{_interpret(best_row['js_div'])}**.")
    lines.append(f"- Curve overlap at best fit: **OVL = {cs_best['overlap']:.3f}** "
                 f"(the two density curves share {cs_best['overlap']:.0%} of their area), "
                 f"Hellinger {cs_best['hellinger']:.3f}, total variation {cs_best['tv']:.3f}.\n")

    lines.append("## Best configurations across the whole sweep\n")
    lines.append("Curve overlap (OVL) and Pearson shape-correlation are computed on the density "
                 "curves for **every** (forward, lateral) cell of the sweep; the table lists the "
                 "eight closest to the real distribution, ranked by overlap. The ★ row is the "
                 "overall best fit (minimum Wasserstein).\n")
    top = sweep.sort_values("overlap", ascending=False).head(8).reset_index(drop=True)
    tcols = ["", "forward", "lateral", "Overlap ↑", "Pearson ↑", "Wass. ↓ (m/s)", "sim mean", "sim SD"]
    lines.append("| " + " | ".join(tcols) + " |")
    lines.append("|" + "|".join(["---"] * len(tcols)) + "|")
    for _, r in top.iterrows():
        is_best = np.isclose(r["forward"], best["forward"]) and np.isclose(r["lateral"], best["lateral"])
        b = "**" if is_best else ""
        lines.append("| " + " | ".join([
            "★" if is_best else "",
            f"{b}{r['forward']:.3f}{b}", f"{b}{r['lateral']:.2f}{b}",
            f"{b}{r['overlap']:.3f}{b}", f"{r['pearson']:.3f}",
            f"{r['wasserstein']:.4f}", f"{r['mean_sim']:.3f}", f"{r['std_sim']:.3f}",
        ]) + " |")
    lines.append("")

    lines.append("## Statistical battery (representative forward coefficients, "
                 f"lateral={best['lateral']:.2f})\n")
    cols = ["Label", "N sim", "Wass. (m/s)", "Energy", "JS div", "sim mean", "Δmean",
            "sim SD", "Cohen d", "KS stat", "KS p", "AD p", "MW p"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for lab, r in zip(labels, stat_rows):
        lines.append("| " + " | ".join([
            lab, f"{r['n_sim']}", f"{r['wasserstein']:.4f}", f"{r['energy']:.4f}",
            f"{r['js_div']:.3f}", f"{r['mean_sim']:.3f}", f"{r['d_mean']:+.3f}",
            f"{r['std_sim']:.3f}", f"{r['cohens_d']:+.2f}", f"{r['ks_stat']:.3f}",
            f"{r['ks_p']:.1e}", f"{r['ad_p']:.3f}", f"{r['mw_p']:.1e}",
        ]) + " |")
    lines.append("")

    lines.append("## Curve-shape similarity (on the density curves themselves)\n")
    lines.append("Metrics computed directly on the KDE curves f_real, f_sim over a shared "
                 "grid — this is what the eye compares when the curves are overlaid. "
                 "Overlap/Pearson/Cosine: **higher = more similar** (1 = identical). "
                 "Hellinger/TV/Bhattacharyya/Jeffreys: **lower = more similar** (0 = identical).\n")
    scols = ["Label", "Overlap ↑", "Pearson ↑", "Cosine ↑", "Hellinger ↓", "TV ↓",
             "Bhattach. ↓", "Jeffreys ↓"]
    lines.append("| " + " | ".join(scols) + " |")
    lines.append("|" + "|".join(["---"] * len(scols)) + "|")
    for lab, r in zip(labels, stat_rows):
        lines.append("| " + " | ".join([
            lab, f"{r['overlap']:.3f}", f"{r['pearson']:.3f}", f"{r['cosine']:.3f}",
            f"{r['hellinger']:.3f}", f"{r['tv']:.3f}", f"{r['bhattacharyya']:.3f}",
            f"{r['jeffreys']:.3f}",
        ]) + " |")
    lines.append("")

    lines.append("## Remaining discrepancy\n")
    lines += recommendations(real_v, sweep, best)

    with open(os.path.join(OUT_DIR, "analysis_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_latex_tables(real_v, sweep, best, stat_rows, labels):
    """Emit the headline tables as booktabs LaTeX, ready to \\input into the paper.

    Requires \\usepackage{booktabs} in the document preamble.
    """
    out = [r"% Auto-generated by compare_distributions.py — needs \usepackage{booktabs}", ""]

    # --- Table 1: best configurations across the whole sweep, ranked by overlap.
    top = sweep.sort_values("overlap", ascending=False).head(8).reset_index(drop=True)
    out += [
        r"\begin{table}[t]", r"  \centering",
        r"  \caption{Simulation configurations closest to the real (KABR) velocity "
        r"distribution, ranked by curve overlap (OVL). Overlap and Pearson shape-correlation "
        r"are computed on the density curves for every cell of the parameter sweep; the "
        r"$\star$ row is the overall best fit (minimum Wasserstein distance $W$).}",
        r"  \label{tab:best-configs}",
        r"  \begin{tabular}{llcccccc}", r"    \toprule",
        r"    & Forward & Lateral & OVL\,$\uparrow$ & Pearson\,$\uparrow$ & "
        r"$W$\,$\downarrow$ (m/s) & Mean & SD \\", r"    \midrule",
    ]
    for _, r in top.iterrows():
        is_best = np.isclose(r["forward"], best["forward"]) and np.isclose(r["lateral"], best["lateral"])
        bold = (lambda s: r"\textbf{" + s + "}") if is_best else (lambda s: s)
        out.append("    " + " & ".join([
            r"$\star$" if is_best else "",
            bold(f"{r['forward']:.3f}"), bold(f"{r['lateral']:.2f}"),
            bold(f"{r['overlap']:.3f}"), f"{r['pearson']:.3f}",
            f"{r['wasserstein']:.4f}", f"{r['mean_sim']:.3f}", f"{r['std_sim']:.3f}",
        ]) + r" \\")
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    # --- Table 2: curve-shape similarity for the representative configurations.
    out += [
        r"\begin{table}[t]", r"  \centering",
        r"  \caption{Curve-shape similarity between the real and simulated velocity densities "
        r"at representative forward coefficients (lateral $=" + f"{best['lateral']:.2f}" + r"$). "
        r"$\uparrow$/$\downarrow$ mark the direction of closer agreement.}",
        r"  \label{tab:curve-similarity}",
        r"  \begin{tabular}{lccccc}", r"    \toprule",
        r"    Configuration & OVL\,$\uparrow$ & Pearson\,$\uparrow$ & "
        r"Hellinger\,$\downarrow$ & TV\,$\downarrow$ & $W$\,$\downarrow$ (m/s) \\", r"    \midrule",
    ]
    for lab, r in zip(labels, stat_rows):
        name = lab.replace("★", r"$\star$")
        out.append("    " + " & ".join([
            name, f"{r['overlap']:.3f}", f"{r['pearson']:.3f}",
            f"{r['hellinger']:.3f}", f"{r['tv']:.3f}", f"{r['wasserstein']:.4f}",
        ]) + r" \\")
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    with open(os.path.join(OUT_DIR, "tables.tex"), "w") as f:
        f.write("\n".join(out) + "\n")


def recommendations(real_v, sweep, best):
    """Data-driven tuning advice derived from the sweep geometry.

    The advice is grounded in what each parameter actually *does* in the sweep,
    measured directly, rather than assumed:
      * how the simulated mean and SD respond to `intrinsicForwardCoefficient`, and
      * how they respond to `intrinsicLateralMultiplier`.
    """
    real_mean = real_v.mean()
    real_std = real_v.std(ddof=1)
    real_med = np.median(real_v)
    real_skew = float(stats.skew(real_v))
    rec = []
    lat_grid = sorted(sweep["lateral"].unique())
    fwd_grid = sorted(sweep["forward"].unique())

    # --- Measure the response of mean/SD to each parameter, from the sweep itself.
    # forward: correlation of mean & SD with forward, at the best lateral.
    g_lat = sweep[sweep["lateral"] == best["lateral"]].sort_values("forward")
    corr_mean_fwd = np.corrcoef(g_lat["forward"], g_lat["mean_sim"])[0, 1]
    corr_std_fwd = np.corrcoef(g_lat["forward"], g_lat["std_sim"])[0, 1]
    # lateral: how mean & SD change across lateral, at the best forward.
    g_fwd = sweep.iloc[(sweep["forward"] - best["forward"]).abs().groupby(sweep["lateral"]).idxmin()]
    lat_mean_spread = g_fwd["mean_sim"].max() - g_fwd["mean_sim"].min()
    lat_std_spread = g_fwd["std_sim"].max() - g_fwd["std_sim"].min()

    # --- 1. Mean.
    if best["mean_sim"] > real_mean * 1.05:
        rec.append(f"- **Animals move too fast** (sim mean {best['mean_sim']:.3f} vs real "
                   f"{real_mean:.3f} m/s): lower `intrinsicForwardCoefficient`.")
    elif best["mean_sim"] < real_mean * 0.95:
        rec.append(f"- **Animals move too slowly** (sim mean {best['mean_sim']:.3f} vs real "
                   f"{real_mean:.3f} m/s): raise `intrinsicForwardCoefficient`.")
    else:
        rec.append(f"- **Mean speed already matches** at the best fit "
                   f"(sim {best['mean_sim']:.3f} vs real {real_mean:.3f} m/s).")

    # --- 2. Spread, with the mechanism grounded in the measured parameter responses.
    if best["std_sim"] < real_std * 0.9:
        msg = (f"- **The velocity distribution is too narrow** (sim SD {best['std_sim']:.3f} vs "
               f"real {real_std:.3f} m/s) — this is the main remaining gap. ")
        # Is spread controllable independently of the mean?
        if corr_std_fwd > 0.5 and corr_mean_fwd > 0.5:
            msg += (f"In this model SD and mean are **coupled through "
                    f"`intrinsicForwardCoefficient`** (both rise with it: r={corr_std_fwd:.2f} and "
                    f"r={corr_mean_fwd:.2f}), so you cannot widen the distribution by raising the "
                    f"forward coefficient without also overshooting the mean. ")
        if lat_std_spread < 0.3 * lat_mean_spread:
            msg += (f"`intrinsicLateralMultiplier` mostly **shifts the mean** "
                    f"(Δmean≈{lat_mean_spread:.3f} vs Δspread≈{lat_std_spread:.3f} across the tested "
                    f"lateral values), so it will not supply the missing width either. ")
        msg += ("**The width must come from a new source of variability that the current two "
                "parameters do not provide**, e.g.: (a) heterogeneity in the forward coefficient "
                "*across agents* (draw it from a distribution instead of a single value); "
                "(b) a two-state rest/move (start-stop) dynamic so some agents sit near the floor "
                "while others move fast; or (c) a larger velocity-noise term.")
        rec.append(msg)
    elif best["std_sim"] > real_std * 1.1:
        rec.append(f"- **The velocity distribution is too wide** (sim SD {best['std_sim']:.3f} vs "
                   f"real {real_std:.3f}). Reduce velocity noise / lateral diffusion to concentrate it.")
    else:
        rec.append("- **Velocity spread already matches** at the best fit.")

    # --- 3. Shape / skew.
    rec.append(f"- The real distribution is **right-skewed** (skew {real_skew:.2f}, "
               f"median {real_med:.3f} < mean {real_mean:.3f}) and censored at {FLOOR_MS:.3f} m/s: "
               f"most animals are slow with an occasional fast burst. A heavy right tail (a "
               f"start-stop / bursty speed process) reproduces both the skew and the missing SD at "
               f"once, and is a better lever here than any single global parameter.")

    # --- 4. Boundary warnings -> the grid may be too small.
    if np.isclose(best["forward"], fwd_grid[0]) or np.isclose(best["forward"], fwd_grid[-1]):
        d = "below" if np.isclose(best["forward"], fwd_grid[0]) else "above"
        rec.append(f"- The optimal forward coefficient sits on the **edge of the swept range** "
                   f"[{fwd_grid[0]:.2f}, {fwd_grid[-1]:.2f}]; extend the sweep {d} it to be sure "
                   f"the optimum is interior.")
    if np.isclose(best["lateral"], lat_grid[0]) or np.isclose(best["lateral"], lat_grid[-1]):
        d = "below" if np.isclose(best["lateral"], lat_grid[0]) else "above"
        rec.append(f"- The optimal lateral multiplier is the **{'lowest' if d=='below' else 'highest'} "
                   f"value tested** ({best['lateral']:.2f}); test values {d} the current "
                   f"[{lat_grid[0]:.2f}, {lat_grid[-1]:.2f}] range.")
    return rec


if __name__ == "__main__":
    main()

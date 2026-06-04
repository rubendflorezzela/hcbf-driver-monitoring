"""
analysis.py
===========
Consumes results/raw_results.json and produces:
  - Normalized efficiency score eta (Eq. 4, min-max across models)
  - Final four-dimensional HCBF profile per model
  - Pareto frontier (non-dominated models)
  - Human-Centered Score under the three scenarios (Table 1)
  - LaTeX tables (results/tables/) and radar-chart figure (results/figures/)

Usage:
    python analysis.py

Author: Ruben Dario Florez-Zela
"""

import os
import json
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C


def load_raw():
    with open(os.path.join(C.OUTPUT_DIR, "raw_results.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Efficiency normalization (Eq. 4)
# ---------------------------------------------------------------------------
def compute_efficiency_scores(raw):
    names = list(raw.keys())
    P = np.array([raw[n]["efficiency"]["params_M"] for n in names])
    Fl = np.array([raw[n]["efficiency"]["flops_G"] for n in names])
    L = np.array([raw[n]["efficiency"]["latency_ms"] for n in names])

    def norm(v):
        rng = v.max() - v.min()
        return np.zeros_like(v) if rng == 0 else (v - v.min()) / rng

    eta = 1.0 - (norm(P) + norm(Fl) + norm(L)) / 3.0
    return {n: float(eta[i]) for i, n in enumerate(names)}


# ---------------------------------------------------------------------------
# Build the four-dimensional profile
# ---------------------------------------------------------------------------
def build_profiles(raw):
    eta_scores = compute_efficiency_scores(raw)
    profiles = {}
    for n in raw:
        profiles[n] = {
            "alpha": raw[n]["accuracy"]["alpha"],
            "eps": raw[n].get("explainability", {}).get("eps", float("nan")),
            "eta": eta_scores[n],
            "rho": raw[n]["robustness"]["rho"],
        }
    return profiles


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------
def pareto_frontier(profiles):
    """Return set of non-dominated model names (all four dims, higher=better)."""
    dims = ["alpha", "eps", "eta", "rho"]
    names = list(profiles.keys())
    nondominated = []
    for a in names:
        dominated = False
        for b in names:
            if a == b:
                continue
            ge_all = all(profiles[b][d] >= profiles[a][d] for d in dims)
            gt_any = any(profiles[b][d] > profiles[a][d] for d in dims)
            if ge_all and gt_any:
                dominated = True
                break
        if not dominated:
            nondominated.append(a)
    return nondominated


# ---------------------------------------------------------------------------
# Human-Centered Score (Eq. 6)
# ---------------------------------------------------------------------------
def compute_hcs(profiles):
    rows = []
    for scenario, w in C.HCS_SCENARIOS.items():
        for n, p in profiles.items():
            hcs = (w["alpha"] * p["alpha"] + w["eps"] * p["eps"]
                   + w["eta"] * p["eta"] + w["rho"] * p["rho"])
            rows.append({"scenario": scenario, "model": n, "hcs": hcs})
    df = pd.DataFrame(rows)
    # ranking per scenario
    df["rank"] = df.groupby("scenario")["hcs"].rank(ascending=False, method="min")
    return df


# ---------------------------------------------------------------------------
# Output: Tables
# ---------------------------------------------------------------------------
def write_profile_table(raw, profiles):
    lines = [
        r"\begin{table}[t]",
        r"  \caption{HCBF four-dimensional profiles. All scores normalized to [0,1]; higher is better.}",
        r"  \label{tab:profiles}",
        r"  \centering",
        r"  \begin{tabular}{lcccc}",
        r"    \hline",
        r"    \textbf{Model} & $\alpha$ (Acc) & $\varepsilon$ (Expl) & $\eta$ (Eff) & $\rho$ (Rob) \\",
        r"    \hline",
    ]
    for n in C.MODELS:
        if n not in profiles:
            continue
        p = profiles[n]
        lines.append(
            f"    {C.MODEL_DISPLAY[n]} & {p['alpha']:.3f} & {p['eps']:.3f} "
            f"& {p['eta']:.3f} & {p['rho']:.3f} \\\\"
        )
    lines += [r"    \hline", r"  \end{tabular}", r"\end{table}"]
    path = os.path.join(C.TABLE_DIR, "table_profiles.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def write_hcs_table(hcs_df):
    pivot = hcs_df.pivot(index="model", columns="scenario", values="hcs")
    rank = hcs_df.pivot(index="model", columns="scenario", values="rank")
    lines = [
        r"\begin{table}[t]",
        r"  \caption{Human-Centered Score under three scenarios. Rank in parentheses.}",
        r"  \label{tab:hcs_results}",
        r"  \centering",
        r"  \begin{tabular}{lccc}",
        r"    \hline",
        r"    \textbf{Model} & HCS-A & HCS-B & HCS-C \\",
        r"    \hline",
    ]
    scenarios = list(C.HCS_SCENARIOS.keys())
    for n in C.MODELS:
        if n not in pivot.index:
            continue
        cells = []
        for s in scenarios:
            cells.append(f"{pivot.loc[n, s]:.3f} ({int(rank.loc[n, s])})")
        lines.append(f"    {C.MODEL_DISPLAY[n]} & " + " & ".join(cells) + r" \\")
    lines += [r"    \hline", r"  \end{tabular}", r"\end{table}"]
    path = os.path.join(C.TABLE_DIR, "table_hcs.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Output: radar chart
# ---------------------------------------------------------------------------
def plot_radar(profiles):
    dims = ["Accuracy", "Explainability", "Efficiency", "Robustness"]
    keys = ["alpha", "eps", "eta", "rho"]
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for n in C.MODELS:
        if n not in profiles:
            continue
        vals = [profiles[n][k] for k in keys]
        vals += vals[:1]
        ax.plot(angles, vals, label=C.MODEL_DISPLAY[n], linewidth=1.8)
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=9)
    plt.tight_layout()
    path = os.path.join(C.FIGURE_DIR, "radar_profiles.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    raw = load_raw()
    profiles = build_profiles(raw)

    print("\n=== HCBF profiles (normalized, higher=better) ===")
    for n in C.MODELS:
        if n in profiles:
            p = profiles[n]
            print(f"  {C.MODEL_DISPLAY[n]:20s} "
                  f"alpha={p['alpha']:.3f} eps={p['eps']:.3f} "
                  f"eta={p['eta']:.3f} rho={p['rho']:.3f}")

    pf = pareto_frontier(profiles)
    print("\n=== Pareto frontier (non-dominated) ===")
    print("  " + ", ".join(C.MODEL_DISPLAY[n] for n in pf))

    hcs_df = compute_hcs(profiles)
    print("\n=== Human-Centered Score ===")
    print(hcs_df.pivot(index="model", columns="scenario",
                       values="hcs").round(3).to_string())

    # Ranking stability check
    ranks = hcs_df.pivot(index="model", columns="scenario", values="rank")
    top_per_scenario = ranks.idxmin()
    stable = len(set(top_per_scenario.values)) == 1
    print(f"\nTop model identical across all scenarios: {stable}")
    if stable:
        print(f"  Winner: {C.MODEL_DISPLAY[top_per_scenario.iloc[0]]}")

    t1 = write_profile_table(raw, profiles)
    t2 = write_hcs_table(hcs_df)
    fig = plot_radar(profiles)
    print(f"\nWrote: {t1}\n       {t2}\n       {fig}")

    summary = {
        "profiles": profiles,
        "pareto_frontier": pf,
        "hcs": hcs_df.to_dict(orient="records"),
        "ranking_stable": bool(stable),
    }
    with open(os.path.join(C.OUTPUT_DIR, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

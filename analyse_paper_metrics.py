#!/usr/bin/env python3
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

OUTDIR = "analysis_output_ieee_v2"
os.makedirs(OUTDIR, exist_ok=True)

REWARDS = ["PBR", "EFFRES", "IVI", "NNSI"]

METRICS = [
    "λ₂",
    "AvgNodeConn",
    "GCC_5%",
    "ASPL",
    "Diameter",
    "ArticulationPoints",
    "Bridges",
    "BetCentralization",
    "NatConnectivity",
    "EffResistance",
    "Assortativity",
    "AvgClustering",
]

PRIMARY_METRICS = ["λ₂", "EffResistance", "GCC_5%", "ASPL", "AvgNodeConn", "NatConnectivity"]

HIGHER_BETTER = {
    "λ₂": True,
    "AvgNodeConn": True,
    "NatConnectivity": True,
    "GCC_5%": True,
    "ASPL": False,
    "Diameter": False,
    "ArticulationPoints": False,
    "Bridges": False,
    "BetCentralization": False,
    "EffResistance": False,
    "Assortativity": True,   # interpret carefully
    "AvgClustering": True,
}

EFFRES_PENALTY = 1e9  # used in your main script for disconnected cases

def latex_escape(s: str) -> str:
    """Minimal LaTeX escaping for table content."""
    if not isinstance(s, str):
        s = str(s)
    return (s.replace("\\", "\\textbackslash{}")
             .replace("%", "\\%")
             .replace("_", "\\_")
             .replace("&", "\\&")
             .replace("#", "\\#")
             .replace("$", "\\$")
             .replace("{", "\\{")
             .replace("}", "\\}")
             .replace("~", "\\textasciitilde{}")
             .replace("^", "\\textasciicircum{}"))

def df_to_ieee_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    column_order: list[str],
    column_headers: dict[str, str],
    align: str = None
) -> str:
    """
    Convert a DataFrame to an IEEE-friendly LaTeX table.

    - column_order: list of columns from df to include in order
    - column_headers: mapping col -> displayed header (can include LaTeX math)
    - align: tabular alignment string; if None, uses 'l' + 'c' repeated
    """
    use = df[column_order].copy()

    # Default alignment: left for first col, centered for rest
    if align is None:
        align = "l" + ("c" * (len(column_order) - 1))

    # Header row
    header_cells = []
    for col in column_order:
        header_cells.append(column_headers.get(col, latex_escape(col)))
    header = " & ".join(header_cells) + " \\\\"

    # Body rows
    body_lines = []
    for _, row in use.iterrows():
        cells = []
        for col in column_order:
            val = row[col]
            # Keep LaTeX symbols like λ₂ as-is in headers only, in body we store numeric strings
            cells.append(latex_escape(val) if isinstance(val, str) else latex_escape(str(val)))
        body_lines.append(" & ".join(cells) + " \\\\")

    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append(f"\\begin{{tabular}}{{{align}}}")
    latex.append("\\hline")
    latex.append(header)
    latex.append("\\hline")
    latex.extend(body_lines)
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def delta_col(method: str, metric: str) -> str:
    return f"%Δ_{method}_vs_Orig_{metric}"


def median_iqr(s: pd.Series):
    s = s.dropna()
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    q1 = float(s.quantile(0.25))
    med = float(s.quantile(0.50))
    q3 = float(s.quantile(0.75))
    return med, q1, q3


def holm_correction(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    for k, idx in enumerate(order):
        adj[idx] = min(1.0, (m - k) * pvals[idx])
    # enforce monotonicity
    adj_sorted = adj[order]
    for i in range(1, m):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i - 1])
    adj[order] = adj_sorted
    return adj


def compute_density(df: pd.DataFrame) -> pd.Series:
    return np.where(df["N"] > 1, (2 * df["M"]) / (df["N"] * (df["N"] - 1)), np.nan)


def table_I_dataset_characterization(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["Density"] = compute_density(tmp)

    cols = ["N", "M", "Density"]
    for met in PRIMARY_METRICS + ["Diameter", "ArticulationPoints", "Bridges", "BetCentralization", "Assortativity", "AvgClustering"]:
        c = f"Orig_{met}"
        if c in tmp.columns:
            cols.append(c)

    rows = []
    for c in cols:
        s = tmp[c].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) == 0:
            continue
        rows.append({
            "Metric": c,
            "Min": float(s.min()),
            "Median": float(s.median()),
            "Max": float(s.max()),
            "Mean": float(s.mean()),
            "Std": float(s.std()),
        })
    return pd.DataFrame(rows)


def table_II_main_results(df: pd.DataFrame) -> pd.DataFrame:
    use_metrics = [m for m in PRIMARY_METRICS if any(delta_col(r, m) in df.columns for r in REWARDS)]
    rows = []
    for r in REWARDS:
        row = {"Method": r}
        for met in use_metrics:
            c = delta_col(r, met)
            if c not in df.columns:
                row[met] = ""
                continue
            med, q1, q3 = median_iqr(df[c])
            row[met] = f"{med:.2f} [{q1:.2f}, {q3:.2f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def table_III_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for met in METRICS:
        cols = {r: delta_col(r, met) for r in REWARDS if delta_col(r, met) in df.columns}
        if len(cols) < 2:
            continue
        hb = HIGHER_BETTER.get(met, True)

        sub = df[list(cols.values())].copy()
        sub = sub.replace([np.inf, -np.inf], np.nan)

        if hb:
            best_col = sub.idxmax(axis=1)
        else:
            best_col = sub.idxmin(axis=1)

        best_method = best_col.str.extract(r"%Δ_(.*)_vs_Orig_")[0]
        counts = best_method.value_counts(dropna=True).to_dict()

        for r in REWARDS:
            rows.append({"Metric": met, "Method": r, "Wins": int(counts.get(r, 0))})
    return pd.DataFrame(rows)


def table_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for met in METRICS:
        cols = {r: delta_col(r, met) for r in REWARDS if delta_col(r, met) in df.columns}
        if len(cols) < 2:
            continue

        pairs = list(itertools.combinations(cols.keys(), 2))
        tmp = []
        pvals = []

        for a, b in pairs:
            d = df[[cols[a], cols[b]]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(d) < 5:
                continue

            x = d[cols[a]].values
            y = d[cols[b]].values

            try:
                stat, p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                stat, p = np.nan, np.nan

            tmp.append((met, a, b, len(d), float(np.median(x - y)), float(np.mean(x - y)), stat, p))
            pvals.append(p)

        if not tmp:
            continue

        adj = holm_correction(np.array(pvals, dtype=float))
        for (met, a, b, n, meddiff, meandiff, stat, p), p_adj in zip(tmp, adj):
            rows.append({
                "Metric": met,
                "A": a,
                "B": b,
                "N_graphs": n,
                "MedianDiff(A-B)": meddiff,
                "MeanDiff(A-B)": meandiff,
                "WilcoxonStat": stat,
                "p_value": p,
                "p_value_holm": float(p_adj),
                "Significant_0.05": bool(np.isfinite(p_adj) and p_adj < 0.05),
            })

    return pd.DataFrame(rows)


def table_size_bins(df: pd.DataFrame) -> pd.DataFrame:
    if "N" not in df.columns:
        return pd.DataFrame()

    bins = pd.cut(
        df["N"],
        bins=[0, 40, 93, 10_000],
        labels=["Small (≤40)", "Medium (41–93)", "Large (>93)"]
    )

    rows = []
    for met in ["GCC_5%", "λ₂"]:
        if met not in METRICS:
            continue
        for r in REWARDS:
            c = delta_col(r, met)
            if c not in df.columns:
                continue
            for b in ["Small (≤40)", "Medium (41–93)", "Large (>93)"]:
                s = df.loc[bins == b, c].replace([np.inf, -np.inf], np.nan).dropna()
                med, q1, q3 = median_iqr(s)
                rows.append({
                    "Metric": met,
                    "Method": r,
                    "SizeBin": b,
                    "Median_%Δ": med,
                    "IQR25_%Δ": q1,
                    "IQR75_%Δ": q3,
                    "N_graphs": int(len(s)),
                })
    return pd.DataFrame(rows)


def table_data_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Original columns
    for met in METRICS:
        c = f"Orig_{met}"
        if c not in df.columns:
            continue
        s = df[c]
        rows.append({
            "Scope": "Original",
            "Column": c,
            "NaN_Count": int(s.isna().sum()),
            "Inf_Count": int(np.isinf(s).sum()),
            "Penalty_Count": int((s == EFFRES_PENALTY).sum()) if met == "EffResistance" else 0,
        })

    # Delta columns
    for met in METRICS:
        for r in REWARDS:
            c = delta_col(r, met)
            if c not in df.columns:
                continue
            s = df[c]
            rows.append({
                "Scope": "Delta",
                "Column": c,
                "NaN_Count": int(s.isna().sum()),
                "Inf_Count": int(np.isinf(s).sum()),
                "Penalty_Count": 0,
            })

    return pd.DataFrame(rows)


def fig_1_dataset_correlation_heatmap(df: pd.DataFrame) -> None:
    tmp = df.copy()
    tmp["Density"] = compute_density(tmp)

    cols = ["N", "M", "Density"]
    for met in METRICS:
        c = f"Orig_{met}"
        if c in tmp.columns:
            cols.append(c)

    mat = tmp[cols].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if len(mat) < 5:
        return

    corr = mat.corr(numeric_only=True)

    plt.figure(figsize=(9.0, 7.5))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=8)
    plt.title("Dataset Metric Correlation Heatmap (Original Graphs)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_1_dataset_correlation_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()


def fig_2_improvement_boxplots(df: pd.DataFrame) -> None:
    show = [m for m in ["λ₂", "EffResistance", "GCC_5%", "ASPL"] if m in METRICS]
    show = [m for m in show if any(delta_col(r, m) in df.columns for r in REWARDS)]
    if not show:
        return

    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 3.8))
    if len(show) == 1:
        axes = [axes]

    for ax, met in zip(axes, show):
        data, labels = [], []
        for r in REWARDS:
            c = delta_col(r, met)
            if c not in df.columns:
                continue
            data.append(df[c].replace([np.inf, -np.inf], np.nan).dropna().values)
            labels.append(r)

        ax.boxplot(data, labels=labels, showfliers=False)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(met)
        ax.set_ylabel("%Δ vs. Original")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_2_improvement_boxplots.png"), dpi=300, bbox_inches="tight")
    plt.close()


def fig_3_size_bin_gcc5(df: pd.DataFrame) -> None:
    if "N" not in df.columns:
        return
    met = "GCC_5%"
    if not any(delta_col(r, met) in df.columns for r in REWARDS):
        return

    bins = pd.cut(df["N"], bins=[0, 40, 93, 10_000], labels=["Small", "Medium", "Large"])
    xcats = ["Small", "Medium", "Large"]

    plt.figure(figsize=(7.5, 4.2))
    for r in REWARDS:
        c = delta_col(r, met)
        if c not in df.columns:
            continue
        ys = []
        for b in xcats:
            s = df.loc[bins == b, c].replace([np.inf, -np.inf], np.nan).dropna()
            ys.append(float(s.median()) if len(s) else np.nan)
        plt.plot(xcats, ys, marker="o", linewidth=2, label=r)

    plt.xlabel("Network size bin")
    plt.ylabel("Median %Δ GCC_5% (vs. original)")
    plt.title("Size-bin generalization on targeted-attack robustness (5%)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_3_size_bin_gcc5.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    path = "results_network_metrics.csv"
    if not os.path.isfile(path):
        raise FileNotFoundError("results_network_metrics.csv not found in current directory.")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("results_network_metrics.csv is empty.")

    # Tables
    t1 = table_I_dataset_characterization(df)
    t1.to_csv(os.path.join(OUTDIR, "table_I_dataset_characterization.csv"), index=False)

    t2 = table_II_main_results(df)
    t2.to_csv(os.path.join(OUTDIR, "table_II_main_results_median_iqr.csv"), index=False)

    t3 = table_III_win_rates(df)
    t3.to_csv(os.path.join(OUTDIR, "table_III_win_rates.csv"), index=False)

    ttests = table_statistical_tests(df)
    ttests.to_csv(os.path.join(OUTDIR, "table_statistical_tests_wilcoxon_holm.csv"), index=False)

    tbins = table_size_bins(df)
    tbins.to_csv(os.path.join(OUTDIR, "table_size_bin_summary.csv"), index=False)

    tqual = table_data_quality_checks(df)
    tqual.to_csv(os.path.join(OUTDIR, "table_data_quality_checks.csv"), index=False)

    # Figures
    fig_1_dataset_correlation_heatmap(df)
    fig_2_improvement_boxplots(df)
    fig_3_size_bin_gcc5(df)

    # Report: what to include in paper
    with open(os.path.join(OUTDIR, "ANALYSIS_REPORT_IEEE_V2.txt"), "w") as f:
        f.write("IEEE PAPER OUTPUT PACKAGE (NO GREEDY, NO GCC 2/10)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Graphs analyzed: {len(df)}\n\n")

        f.write("TABLES TO INCLUDE IN PAPER\n")
        f.write("-" * 80 + "\n")
        f.write("Table I  : Dataset Characterization (ITZ subset)\n")
        f.write("          -> table_I_dataset_characterization.csv\n")
        f.write("Table II : Main Results Across Rewards (Median [IQR])\n")
        f.write("          -> table_II_main_results_median_iqr.csv\n")
        f.write("Table III: Win Rates by Metric\n")
        f.write("          -> table_III_win_rates.csv\n\n")

        f.write("FIGURES TO INCLUDE IN PAPER\n")
        f.write("-" * 80 + "\n")
        f.write("Fig. 1: Dataset Metric Correlation Heatmap (Original Graphs)\n")
        f.write("       -> fig_1_dataset_correlation_heatmap.png\n")
        f.write("Fig. 2: Distribution of Improvements Across Rewards (Boxplots)\n")
        f.write("       -> fig_2_improvement_boxplots.png\n")
        f.write("Fig. 3: Size-bin Generalization on GCC_5%\n")
        f.write("       -> fig_3_size_bin_gcc5.png\n\n")

        f.write("DATA QUALITY CHECKS\n")
        f.write("-" * 80 + "\n")
        f.write("NaN/Inf counts and EffResistance penalty hits:\n")
        f.write("  -> table_data_quality_checks.csv\n")

    print(f"Done. Outputs saved to: {OUTDIR}/")

    # --- LaTeX export for Table II (Main Results) ---
    # Read the already-exported CSV to ensure the LaTeX matches what you will use.
    t2_path = os.path.join(OUTDIR, "table_II_main_results_median_iqr.csv")
    t2_df = pd.read_csv(t2_path)

    # Choose columns to include (keep it compact for IEEE)
    col_order = ["Method"]
    for m in ["λ₂", "EffResistance", "GCC_5%", "ASPL", "AvgNodeConn", "NatConnectivity"]:
        if m in t2_df.columns:
            col_order.append(m)

    # Pretty headers (math where helpful)
    col_headers = {
        "Method": "Reward",
        "λ₂": "$\\Delta\\lambda_2$ (\\%)",
        "EffResistance": "$\\Delta R$ (\\%)",
        "GCC_5%": "$\\Delta$ GCC@5\\% (\\%)",
        "ASPL": "$\\Delta$ ASPL (\\%)",
        "AvgNodeConn": "$\\Delta$ ANC (\\%)",
        "NatConnectivity": "$\\Delta$ NC (\\%)",
    }

    latex_str = df_to_ieee_latex_table(
        t2_df,
        caption="Main results across reward formulations, reported as median [IQR] percentage change relative to the original topology across all networks.",
        label="tab:main_results",
        column_order=col_order,
        column_headers=col_headers,
    )

    tex_out = os.path.join(OUTDIR, "table_II_main_results_median_iqr.tex")
    with open(tex_out, "w", encoding="utf-8") as f:
        f.write(latex_str + "\n")


if __name__ == "__main__":
    main()

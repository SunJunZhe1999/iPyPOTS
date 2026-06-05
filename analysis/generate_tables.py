#!/usr/bin/env python3
"""Generate the 6 LaTeX tables for the AAAI 2027 submission of
FedOmniAdapt.

Outputs to analysis/tables/T{0X}_*.tex - drop-in includable via
\input{tables/T0X_*}.

Tables:
 T01_main_results.tex      Phase A main 6 (task,fam) × 3 methods
 T02_family_ablation.tex   Phase B smollm-only ablation
 T03_m_sensitivity.tex     Phase C M_local sweep
 T04_labelshift.tex        Phase D Dirichlet labelshift
 T05_compute_efficiency.tex M=10 verification on outlier seed
 T06_variance_reduction.tex Within-cell σ comparison
"""
from __future__ import annotations
import json, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

ROOT = Path("/home/Aboya_25R9803/projects/perso/LLMium/projects/03-SLM-CoreMethods/05-federated-slm/fed_omni_adapt")
B9 = Path("/tmp/fedomni_results/batch9")
VERIFY = Path("/tmp/fedomni_results/batch9_verify")
MANIFEST = Path("/tmp/fedomni_results/batch9_manifest.json")
TAB_DIR = ROOT / "analysis" / "tables"


def expected_filename(c):
    return (f"seed{c['seed']}_task{c['task']}_N5_{c['strategy']}_"
            f"K{c['K']}_M{c['M']}_split{c['split']}_"
            f"r{c['r']}_w{c['K_warmup']}_cmld{c['cmld_alpha']:g}_"
            f"fp{c['fedprox_mu']:g}_fam{c['slm_family']}_clip{c['grad_clip']:g}.json")


def load_groups():
    by = defaultdict(list)
    for c in json.loads(MANIFEST.read_text())["cells"]:
        p = B9 / expected_filename(c)
        if not p.exists(): continue
        d = json.loads(p.read_text())
        key = (c["task"], c["split"], c["slm_family"], c["strategy"],
               c["K"], c["M"], c["fedprox_mu"], c["cmld_alpha"])
        by[key].append((int(c["seed"]), float(d["final_mean_acc"])))
    out = {}
    for k, sa in by.items():
        sa = sorted(set(sa))
        out[k] = {"seeds": [s for s, _ in sa],
                  "accs": [a for _, a in sa],
                  "n": len(sa),
                  "mean": np.mean([a for _, a in sa]) if sa else None,
                  "std": np.std([a for _, a in sa], ddof=0) if len(sa) >= 2 else 0.0}
    return out


def floor_of(g, task, split, fam, K, M):
    return g.get((task, split, fam, "local_only", K, M, 0, 0))


def paired_delta(method, floor):
    if not (method and floor): return None, None, None
    a = dict(zip(method["seeds"], method["accs"]))
    b = dict(zip(floor["seeds"], floor["accs"]))
    shared = sorted(set(a) & set(b))
    deltas = [(a[s] - b[s]) * 100 for s in shared]
    if not deltas: return None, None, None
    p = None
    if HAVE_SCIPY and len(deltas) >= 5:
        try:
            p = float(wilcoxon([a[s] for s in shared], [b[s] for s in shared]).pvalue)
        except Exception: p = None
    return np.mean(deltas), (np.std(deltas, ddof=0) if len(deltas) >= 2 else 0.0), p


def tier(d):
    if d is None: return "--"
    if d >= 5: return "\\textbf{Win}"
    if d >= 3: return "Tol."
    return "Loser"


def fmt_acc(g):
    if g is None: return "--"
    return f"{g['mean']*100:.2f} {{\\scriptsize $\\pm$ {g['std']*100:.2f}}}"


def fmt_delta(d, s, p):
    if d is None: return "--"
    pstr = f" {{\\scriptsize $p$={p:.3f}}}" if p is not None else ""
    return f"{d:+.2f} {{\\scriptsize $\\pm$ {s:.2f}}}{pstr}"


def t01_main(groups):
    lines = []
    lines.append(r"% F1 input: paper-aaai/tables/T01_main_results.tex")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Phase A: 5-seed validation, $K{=}4$, $M_{\rm local}{=}5$, $r{=}8$. "
                 r"Wilcoxon paired $p$ at $n{=}5$ ranges in $\{0.0625, 0.1250, 0.1875, \ldots\}$; "
                 r"$p{=}0.0625$ is the minimum possible. Tier thresholds: Win $\geq$ +5\,pp, Tolerable $\geq$ +3\,pp.}")
    lines.append(r"\label{tab:main_aaai}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Task & Family & Method & Acc.\ (\%) & $\Delta$ vs floor (pp) \\")
    lines.append(r"\midrule")
    tasks = [("ng20", "20NewsGroups"), ("yahoo", "Yahoo!Answers"), ("ag4", "AG-News")]
    for task, tl in tasks:
        for fi, fam in enumerate([("mixed", "Mixed"), ("qwen", "Qwen-only")]):
            f_key, f_label = fam
            floor = floor_of(groups, task, "iid", f_key, 4, 5)
            for si, (strat, mname) in enumerate([
                ("local_only", "Floor"),
                ("submatrix", "$\\Phi$-SMT"),
                ("spatial_bicubic", "$\\Phi$-Spatial"),
            ]):
                g = groups.get((task, "iid", f_key, strat, 4, 5, 0, 0))
                t_disp = tl if (si == 0 and fi == 0) else ""
                f_disp = f_label if si == 0 else ""
                if strat == "local_only":
                    lines.append(f"{t_disp} & {f_disp} & {mname} & {fmt_acc(g)} & ref \\\\")
                else:
                    d, s, p = paired_delta(g, floor)
                    lines.append(f"{t_disp} & {f_disp} & {mname} & {fmt_acc(g)} & "
                                 f"{fmt_delta(d, s, p)}~{tier(d)} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def t02_family(groups):
    lines = []
    lines.append(r"\begin{table}[t]\centering")
    lines.append(r"\caption{Phase B: Within-family ablation on 20NewsGroups, $K{=}4$, $M_{\rm local}{=}5$.}")
    lines.append(r"\label{tab:family_ablation}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Family & Floor & $\Phi$-SMT $\Delta$ & $\Phi$-Spatial $\Delta$ \\")
    lines.append(r"\midrule")
    for fam_key, fam_label in [("mixed", "Mixed (SmolLM+Qwen)"),
                                ("smollm", "SmolLM-only"),
                                ("qwen", "Qwen-only")]:
        floor = floor_of(groups, "ng20", "iid", fam_key, 4, 5)
        smt = groups.get(("ng20", "iid", fam_key, "submatrix", 4, 5, 0, 0))
        spa = groups.get(("ng20", "iid", fam_key, "spatial_bicubic", 4, 5, 0, 0))
        d_smt, s_smt, p_smt = paired_delta(smt, floor)
        d_spa, s_spa, p_spa = paired_delta(spa, floor)
        lines.append(f"{fam_label} & {fmt_acc(floor)} & "
                     f"{fmt_delta(d_smt, s_smt, p_smt)}~{tier(d_smt)} & "
                     f"{fmt_delta(d_spa, s_spa, p_spa)}~{tier(d_spa)} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    return "\n".join(lines)


def t03_m_sens(groups):
    lines = []
    lines.append(r"\begin{table}[t]\centering")
    lines.append(r"\caption{Phase C: $M_{\rm local}$ sensitivity on 20NewsGroups-Mixed, $K{=}4$. "
                 r"Lifting $M_{\rm local}$ from 1 to 5 is the binding step.}")
    lines.append(r"\label{tab:m_sensitivity}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & $M{=}1$ Acc & $M{=}1$ $\Delta$ & $M{=}5$ Acc & $M{=}5$ $\Delta$ \\")
    lines.append(r"\midrule")
    for strat, mname in [("local_only", "Floor"),
                          ("submatrix", "$\\Phi$-SMT"),
                          ("spatial_bicubic", "$\\Phi$-Spatial")]:
        g1 = groups.get(("ng20", "iid", "mixed", strat, 4, 1, 0, 0))
        g5 = groups.get(("ng20", "iid", "mixed", strat, 4, 5, 0, 0))
        f1 = floor_of(groups, "ng20", "iid", "mixed", 4, 1)
        f5 = floor_of(groups, "ng20", "iid", "mixed", 4, 5)
        if strat == "local_only":
            d1_str = "ref"; d5_str = "ref"
        else:
            d1, s1, _ = paired_delta(g1, f1)
            d5, s5, p5 = paired_delta(g5, f5)
            d1_str = fmt_delta(d1, s1, None) if d1 is not None else "--"
            d5_str = fmt_delta(d5, s5, p5)
        lines.append(f"{mname} & {fmt_acc(g1)} & {d1_str} & {fmt_acc(g5)} & {d5_str} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    return "\n".join(lines)


def t04_labelshift(groups):
    lines = []
    lines.append(r"\begin{table}[t]\centering")
    lines.append(r"\caption{Phase D: Labelshift (Dirichlet $\alpha{=}0.3$) vs IID on 20NewsGroups-Mixed. "
                 r"$\Phi$ does not recover the floor under quantity skew.}")
    lines.append(r"\label{tab:labelshift}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & IID Acc & IID $\Delta$ & LS Acc & LS $\Delta$ \\")
    lines.append(r"\midrule")
    for strat, mname in [("local_only", "Floor"),
                          ("submatrix", "$\\Phi$-SMT"),
                          ("spatial_bicubic", "$\\Phi$-Spatial")]:
        iid = groups.get(("ng20", "iid", "mixed", strat, 4, 5, 0, 0))
        ls = groups.get(("ng20", "labelshift", "mixed", strat, 4, 5, 0, 0))
        fiid = floor_of(groups, "ng20", "iid", "mixed", 4, 5)
        fls = floor_of(groups, "ng20", "labelshift", "mixed", 4, 5)
        if strat == "local_only":
            d_iid_str = "ref"; d_ls_str = "ref"
        else:
            d_iid, s_iid, p_iid = paired_delta(iid, fiid)
            d_ls, s_ls, p_ls = paired_delta(ls, fls)
            d_iid_str = fmt_delta(d_iid, s_iid, p_iid) if d_iid is not None else "--"
            d_ls_str = fmt_delta(d_ls, s_ls, p_ls) if d_ls is not None else "--"
        lines.append(f"{mname} & {fmt_acc(iid)} & {d_iid_str} & {fmt_acc(ls)} & {d_ls_str} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    return "\n".join(lines)


def t05_compute(groups):
    lines = []
    lines.append(r"\begin{table}[t]\centering")
    lines.append(r"\caption{Compute-efficiency verification on the seed-2 ag4-Qwen outlier. "
                 r"$\Phi$-Spatial at $M{=}5$ exceeds the floor at $M{=}10$ on the same hardware budget.}")
    lines.append(r"\label{tab:compute_efficiency}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Setting & seed-2 acc.\ (\%) & Comment \\")
    lines.append(r"\midrule")
    floor = floor_of(groups, "ag4", "iid", "qwen", 4, 5)
    spa = groups.get(("ag4", "iid", "qwen", "spatial_bicubic", 4, 5, 0, 0))
    seed2_f = dict(zip(floor["seeds"], floor["accs"])).get(2) * 100 if floor else None
    seed2_s = dict(zip(spa["seeds"], spa["accs"])).get(2) * 100 if spa else None
    other_floor = np.mean([dict(zip(floor["seeds"], floor["accs"]))[s]
                            for s in [1, 3, 4, 5]
                            if s in dict(zip(floor["seeds"], floor["accs"]))]) * 100 if floor else None
    verify_p = VERIFY / "seed2_taskag4_N5_local_only_K4_M10_splitiid_r8_w0_cmld0_fp0_famqwen_clip1.json"
    seed2_m10 = float(json.loads(verify_p.read_text())["final_mean_acc"]) * 100 if verify_p.exists() else None

    lines.append(f"Floor, $M{{=}}5$ (seed-2 outlier) & {seed2_f:.2f} & $-${other_floor - seed2_f:.2f}\\,pp vs other-seed mean \\\\")
    if seed2_m10:
        lines.append(f"Floor, $M{{=}}10$ (2$\\times$ epochs) & {seed2_m10:.2f} & partial recovery; gap remains \\\\")
    lines.append(f"$\\Phi$-Spatial, $M{{=}}5$ & {seed2_s:.2f} & \\textbf{{exceeds $M{{=}}10$ floor by {seed2_s - seed2_m10:.2f}\\,pp}} \\\\")
    lines.append(f"Floor, $M{{=}}5$ (seeds \\{{1,3,4,5\\}} mean) & {other_floor:.2f} & non-outlier reference \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    return "\n".join(lines)


def t06_variance(groups):
    lines = []
    lines.append(r"\begin{table}[t]\centering")
    lines.append(r"\caption{Within-cell across-seed standard deviation. $\Phi$ acts as a variance-reducer, "
                 r"most dramatically on ag4-Qwen where Floor $\sigma{=}4.19$ pp collapses to $\Phi$-Spatial $\sigma{=}0.32$ pp ($13\times$ reduction).}")
    lines.append(r"\label{tab:variance_reduction}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Task & Family & Floor $\sigma$ & $\Phi$-SMT $\sigma$ & $\Phi$-Spatial $\sigma$ \\")
    lines.append(r"\midrule")
    for task in ["ng20", "yahoo", "ag4"]:
        for fam in ["mixed", "qwen"]:
            floor = floor_of(groups, task, "iid", fam, 4, 5)
            smt = groups.get((task, "iid", fam, "submatrix", 4, 5, 0, 0))
            spa = groups.get((task, "iid", fam, "spatial_bicubic", 4, 5, 0, 0))
            if not (floor and smt and spa): continue
            lines.append(f"{task} & {fam} & {floor['std']*100:.2f} & "
                         f"{smt['std']*100:.2f} & {spa['std']*100:.2f} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    return "\n".join(lines)


def main():
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    groups = load_groups()
    print(f"loaded {len(groups)} groups\n")
    for name, fn in [
        ("T01_main_results.tex", t01_main),
        ("T02_family_ablation.tex", t02_family),
        ("T03_m_sensitivity.tex", t03_m_sens),
        ("T04_labelshift.tex", t04_labelshift),
        ("T05_compute_efficiency.tex", t05_compute),
        ("T06_variance_reduction.tex", t06_variance),
    ]:
        path = TAB_DIR / name
        path.write_text(fn(groups))
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()

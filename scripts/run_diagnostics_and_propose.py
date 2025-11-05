# kapitel: start
from __future__ import annotations
import sys
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import math
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
try:
    from scipy.stats import gaussian_kde
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def chain_name_from_path(p: Path) -> str:
    return p.stem

def progress_path_for_chain(p: Path) -> Optional[Path]:
    base = p.name
    if not base.endswith('.txt'):
        return None
    prefix = base.split('.')[0]
    prog = p.parent / f'{prefix}.progress'
    return prog if prog.exists() else None

def read_chain_txt(path: Path, mnu_name: str='mnu') -> Tuple[np.ndarray, np.ndarray]:
    values: List[float] = []
    weights: List[float] = []
    weight_idx = None
    mnu_idx = None
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith('#'):
                cols = line[1:].strip().split()
                try:
                    weight_idx = cols.index('weight')
                except ValueError:
                    weight_idx = 0
                if mnu_name in cols:
                    mnu_idx = cols.index(mnu_name)
                elif mnu_name == 'mnu' and 'm_ncdm' in cols:
                    mnu_idx = cols.index('m_ncdm')
                continue
            parts = line.split()
            if weight_idx is None or mnu_idx is None:
                continue
            try:
                w = float(parts[weight_idx])
                v = float(parts[mnu_idx])
            except (IndexError, ValueError):
                continue
            weights.append(w)
            values.append(v)
    if not values:
        return (np.array([]), np.array([]))
    return (np.asarray(values, dtype=float), np.asarray(weights, dtype=float))

def apply_burn_in(values: np.ndarray, weights: np.ndarray, frac: float) -> Tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return (values, weights)
    n = len(values)
    k = int(n * frac)
    return (values[k:], weights[k:])

def apply_trim(values: np.ndarray, weights: np.ndarray, lo: Optional[float]=None, hi: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return (values, weights)
    lo_v = -np.inf if lo is None else lo
    hi_v = np.inf if hi is None else hi
    m = (values >= lo_v) & (values <= hi_v)
    return (values[m], weights[m])

def wsum(w: np.ndarray) -> float:
    return float(np.sum(w))

def wmean(x: np.ndarray, w: np.ndarray) -> float:
    W = wsum(w)
    if W == 0:
        return float('nan')
    return float(np.sum(x * w) / W)

def wvar(x: np.ndarray, w: np.ndarray) -> float:
    W = wsum(w)
    if W == 0:
        return float('nan')
    mu = wmean(x, w)
    return float(np.sum(w * (x - mu) ** 2) / W)

def wstd(x: np.ndarray, w: np.ndarray) -> float:
    v = wvar(x, w)
    return float(np.sqrt(v))

def wpercentiles(x: np.ndarray, w: np.ndarray, ps: List[float]) -> List[float]:
    if x.size == 0:
        return [float('nan') for _ in ps]
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    W = wsum(ws)
    c = np.cumsum(ws)
    out = []
    for p in ps:
        target = p * W
        idx = np.searchsorted(c, target)
        idx = min(idx, len(xs) - 1)
        out.append(float(xs[idx]))
    return out

def whpd_interval(x: np.ndarray, w: np.ndarray, mass: float) -> Tuple[float, float]:
    if x.size == 0:
        return (float('nan'), float('nan'))
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    W = wsum(ws)
    c = np.cumsum(ws)
    target = mass * W
    i = 0
    j = 0
    best_i = 0
    best_j = 0
    best_width = float('inf')
    while i < len(xs):
        while j < len(xs) and c[j] - (c[i - 1] if i > 0 else 0.0) < target:
            j += 1
        if j >= len(xs):
            break
        width = xs[j] - xs[i]
        if width < best_width:
            best_width = width
            best_i, best_j = (i, j)
        i += 1
    return (float(xs[best_i]), float(xs[best_j]))

def prob_above(x: np.ndarray, w: np.ndarray, thr: float) -> float:
    if x.size == 0:
        return float('nan')
    W = wsum(w)
    if W == 0:
        return float('nan')
    return float(np.sum(w[x > thr]) / W)

def autocorr_fft(x: np.ndarray, max_lag: int=500) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n == 0:
        return np.zeros(max_lag + 1)
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    fx = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(fx * np.conj(fx), n=nfft)[:n]
    acf = acf / acf[0]
    return acf[:max_lag + 1]

def integrated_autocorr_time(x: np.ndarray, max_lag: int=500) -> float:
    acf = autocorr_fft(x, max_lag=max_lag)
    positive = acf[1:]
    if positive.size == 0:
        return 1.0
    cutoff = np.argmax(positive < 0)
    if cutoff == 0 and positive[0] >= 0:
        cutoff = len(positive)
    s = np.sum(positive[:cutoff])
    tau_int = 1.0 + 2.0 * float(s)
    return max(tau_int, 1.0)

def split_rhat(x: np.ndarray, splits: int=4) -> float:
    n = len(x)
    if n < splits * 10:
        return float('nan')
    m = splits
    L = n // m
    segments = [x[i * L:(i + 1) * L] for i in range(m)]
    means = np.array([np.mean(s) for s in segments])
    variances = np.array([np.var(s, ddof=1) for s in segments])
    W = np.mean(variances)
    B = L * np.var(means, ddof=1)
    var_hat = (L - 1) / L * W + B / L
    if W <= 0:
        return float('nan')
    Rhat = np.sqrt(var_hat / W)
    return float(Rhat)

def plot_trace(ax, x):
    ax.plot(np.arange(len(x)), x, lw=0.6, color='tab:blue')
    ax.set_xlabel('step')
    ax.set_ylabel('mnu')
    ax.set_title('Trace')

def plot_hist_kde(ax, x, w=None):
    ax.hist(x, bins=60, density=True, alpha=0.5, color='tab:blue', label='hist')
    try:
        if HAVE_SCIPY and len(x) > 2:
            kde = gaussian_kde(x)
            xs = np.linspace(min(x), max(x), 400)
            ax.plot(xs, kde(xs), color='tab:orange', label='KDE')
    except Exception:
        pass
    ax.set_xlabel('mnu [eV]')
    ax.set_ylabel('density')
    ax.set_title('Histogram + KDE')
    ax.legend(loc='best')

def plot_acf(ax, x):
    acf = autocorr_fft(np.asarray(x), max_lag=500)
    ax.stem(np.arange(len(acf)), acf, linefmt='C0-', markerfmt='C0o', basefmt='k-')
    ax.set_xlim(0, 500)
    ax.set_xlabel('lag')
    ax.set_ylabel('ACF')
    ax.set_title('Autocorrelation (<=500)')

def plot_cdf(ax, x, w=None):
    order = np.argsort(x)
    xs = x[order]
    if w is None:
        ys = np.linspace(0, 1, len(xs), endpoint=False)
    else:
        ws = w[order]
        W = wsum(ws)
        ys = np.cumsum(ws) / (W if W > 0 else 1.0)
    ax.plot(xs, ys, color='tab:green')
    ax.set_xlabel('mnu [eV]')
    ax.set_ylabel('CDF')
    ax.set_title('Empirical CDF')

def read_progress_acceptance_and_r(prog_path: Path) -> Tuple[Optional[float], Optional[float]]:
    if prog_path is None or not prog_path.exists():
        return (None, None)
    acc: Optional[float] = None
    rminus1: Optional[float] = None
    with prog_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    acc = float(parts[2])
                except Exception:
                    pass
                try:
                    rminus1 = float(parts[3]) if parts[3].lower() != 'nan' else None
                except Exception:
                    pass
    return (acc, rminus1)

def analyze_chain(chain_path: Path, outdir_results: Path, outdir_plots: Path, burnin_frac: float, trim_floor: Optional[float]) -> Dict:
    values, weights = read_chain_txt(chain_path)
    raw_n = len(values)
    v_b, w_b = apply_burn_in(values, weights, burnin_frac)
    mu = wmean(v_b, w_b)
    sd = wstd(v_b, w_b)
    pcts = [0.01, 0.05, 0.16, 0.5, 0.84, 0.95, 0.99]
    percs = wpercentiles(v_b, w_b, pcts)
    hpd68 = whpd_interval(v_b, w_b, 0.68)
    hpd95 = whpd_interval(v_b, w_b, 0.95)
    p_gt_006 = prob_above(v_b, w_b, 0.06)
    p_gt_01 = prob_above(v_b, w_b, 0.1)
    x_t = v_b
    tau_int = integrated_autocorr_time(x_t, max_lag=500)
    N = len(x_t)
    N_eff = float(N / tau_int) if tau_int > 0 else float('nan')
    rhat = split_rhat(x_t, splits=4)
    acc_rate, r_prog = read_progress_acceptance_and_r(progress_path_for_chain(chain_path))
    cname = chain_name_from_path(chain_path)
    fig, ax = plt.subplots(figsize=(8, 3))
    plot_trace(ax, x_t)
    fig.tight_layout()
    fig.savefig(outdir_plots / f'{cname}_trace.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_hist_kde(ax, x_t)
    fig.tight_layout()
    fig.savefig(outdir_plots / f'{cname}_hist.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3))
    plot_acf(ax, x_t)
    fig.tight_layout()
    fig.savefig(outdir_plots / f'{cname}_acf.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_cdf(ax, v_b, w_b)
    fig.tight_layout()
    fig.savefig(outdir_plots / f'{cname}_cdf.png', dpi=150)
    plt.close(fig)
    trimmed_stats = None
    if trim_floor is not None:
        v_t, w_t = apply_trim(v_b, w_b, lo=trim_floor, hi=None)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(v_b, bins=60, density=True, alpha=0.4, color='tab:blue', label='untrimmed')
        ax.hist(v_t, bins=60, density=True, alpha=0.4, color='tab:orange', label=f'trimmed >= {trim_floor}')
        ax.set_xlabel('mnu [eV]')
        ax.set_ylabel('density')
        ax.set_title('Untrimmed vs Trimmed')
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(outdir_plots / f'{cname}_compare_trimmed_untrimmed.png', dpi=150)
        plt.close(fig)
        trimmed_stats = {'mean': wmean(v_t, w_t), 'std': wstd(v_t, w_t), 'p50': wpercentiles(v_t, w_t, [0.5])[0] if len(v_t) else float('nan')}
    summary = {'file': str(chain_path), 'chain_name': cname, 'raw_samples': raw_n, 'N': N, 'mean': mu, 'median': percs[pcts.index(0.5)], 'std': sd, 'p01': percs[pcts.index(0.01)], 'p05': percs[pcts.index(0.05)], 'p16': percs[pcts.index(0.16)], 'p50': percs[pcts.index(0.5)], 'p84': percs[pcts.index(0.84)], 'p95': percs[pcts.index(0.95)], 'p99': percs[pcts.index(0.99)], 'hpd68_lo': hpd68[0], 'hpd68_hi': hpd68[1], 'hpd95_lo': hpd95[0], 'hpd95_hi': hpd95[1], 'P_gt_0.06': p_gt_006, 'P_gt_0.1': p_gt_01, 'tau_int': tau_int, 'N_eff': N_eff, 'Rhat_split4': rhat, 'accept_rate_log': acc_rate, 'Rminus1_log': r_prog, 'trimmed_mean': trimmed_stats['mean'] if trimmed_stats else None, 'trimmed_std': trimmed_stats['std'] if trimmed_stats else None, 'trimmed_p50': trimmed_stats['p50'] if trimmed_stats else None}
    return summary

def write_summary_csv_md(summaries: List[Dict], outdir_results: Path):
    csv_path = outdir_results / 'summary_mnu_chains.csv'
    keys = ['file', 'chain_name', 'raw_samples', 'N', 'mean', 'median', 'std', 'p01', 'p05', 'p16', 'p50', 'p84', 'p95', 'p99', 'hpd68_lo', 'hpd68_hi', 'hpd95_lo', 'hpd95_hi', 'P_gt_0.06', 'P_gt_0.1', 'tau_int', 'N_eff', 'Rhat_split4', 'accept_rate_log', 'Rminus1_log', 'trimmed_mean', 'trimmed_std', 'trimmed_p50']
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k, '') for k in keys})
    md_path = outdir_results / 'summary_mnu_chains.md'
    lines = ['# Summary mnu chains', '', 'Getrimmte Posterioren sind bedingt durch prior/floor — enge Unsicherheiten können artefaktbedingt sein. Keine Detektionsaussage, bevor Mock-Recovery & Konvergenz-Checks abgeschlossen sind.', '']
    for s in summaries:
        lines += [f"## {s['chain_name']}", f"- N (post burn-in): {s['N']}", f"- mean ± σ: {s['mean']:.6g} ± {s['std']:.3g}", f"- median [16%,84%]: {s['p50']:.6g} [{s['p16']:.6g}, {s['p84']:.6g}]", f"- HPD68: [{s['hpd68_lo']:.6g}, {s['hpd68_hi']:.6g}], HPD95: [{s['hpd95_lo']:.6g}, {s['hpd95_hi']:.6g}]", f"- P(mnu>0.06)={s['P_gt_0.06']:.3f}, P(mnu>0.1)={s['P_gt_0.1']:.3f}", f"- tau_int≈{s['tau_int']:.2f}, N_eff≈{s['N_eff']:.0f}, R̂_split4={s['Rhat_split4']:.3f}", f"- accept_rate (log): {(s['accept_rate_log'] if s['accept_rate_log'] is not None else 'NA')}", '']
    with md_path.open('w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def write_diagnostics_csv(summaries: List[Dict], outdir_results: Path):
    keys = ['file', 'N', 'tau_int', 'N_eff', 'Rhat_split4', 'accept_rate_log']
    path = outdir_results / 'diagnostics.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k, '') for k in keys})

def build_suggested_yaml(summaries: List[Dict], outdir_analysis: Path):
    taus = [s['tau_int'] for s in summaries if isinstance(s.get('tau_int'), (int, float)) and (not math.isnan(s['tau_int']))]
    tau_ref = float(np.median(taus)) if taus else 50.0
    targets = [2000, 10000]
    suggestions = {}
    for neff in targets:
        n_total = int(math.ceil(neff * tau_ref))
        per_chain = int(math.ceil(n_total / 4))
        suggestions[neff] = {'tau_ref': tau_ref, 'chains': 4, 'steps_per_chain': per_chain}
    content = {'theory': 'camb (unchanged)', 'sampler': 'mcmc', 'proposal_scale': 0.5, 'learn_proposal': True, 'min_accepted_steps': 20000, 'recommendations': suggestions, 'notes': ['Start 4 independent chains with seeds and diverse mnu starts: [0.005, 0.06, 0.1, 0.2] eV', 'Aim acceptance rate 0.2–0.4; adjust proposal_scale accordingly', 'If hard floor 0.059: consider reparam y = log(mnu - 0.059 + 1e-6) and sample y']}
    p = outdir_analysis / 'suggested_cobaya.yaml'
    with p.open('w', encoding='utf-8') as f:
        f.write('# Suggested Cobaya sampler overrides (review required)\n')
        f.write('# Please integrate manually into your YAML.\n')
        f.write(json.dumps(content, indent=2))

def write_sample_requirements(summaries: List[Dict], outdir_results: Path):
    path = outdir_results / 'sample_requirements.csv'
    keys = ['file', 'tau_int', 'N_current', 'N_eff_current', 'N_needed_eff2000', 'extra_needed_eff2000', 'N_needed_eff10000', 'extra_needed_eff10000']
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in summaries:
            tau = s['tau_int'] if isinstance(s.get('tau_int'), (int, float)) and (not math.isnan(s['tau_int'])) else float('nan')
            N = s['N']
            Neff = s['N_eff'] if isinstance(s.get('N_eff'), (int, float)) else float('nan')

            def need(target):
                return int(math.ceil(target * tau)) if tau == tau else ''
            n2000 = need(2000)
            n10000 = need(10000)
            row = {'file': s['file'], 'tau_int': tau, 'N_current': N, 'N_eff_current': Neff, 'N_needed_eff2000': n2000, 'extra_needed_eff2000': n2000 - N if isinstance(n2000, int) else '', 'N_needed_eff10000': n10000, 'extra_needed_eff10000': n10000 - N if isinstance(n10000, int) else ''}
            w.writerow(row)

def mh_mock_recovery(mnu_true: float, floor: Optional[float], steps: int=20000, proposal_sigma: float=0.002, seed: int=0) -> Dict:
    rng = np.random.default_rng(seed)
    sigma_like = 0.0015

    def loglike(m):
        if floor is not None and m < floor:
            return -np.inf
        return -0.5 * ((m - mnu_true) / sigma_like) ** 2
    m = max(mnu_true, floor if floor is not None else -np.inf)
    accept = 0
    chain = []
    for i in range(steps):
        prop = m + rng.normal(0, proposal_sigma)
        a = loglike(prop) - loglike(m)
        if np.log(rng.random()) < a:
            m = prop
            accept += 1
        chain.append(m)
    chain = np.array(chain)
    acc_rate = accept / steps
    vals = chain[steps // 10:]
    w = np.ones_like(vals)
    mu = wmean(vals, w)
    hpd68 = whpd_interval(vals, w, 0.68)
    return {'mnu_true': mnu_true, 'mean': mu, 'median': np.median(vals), 'hpd68_lo': hpd68[0], 'hpd68_hi': hpd68[1], 'in_hpd68': mnu_true >= hpd68[0] and mnu_true <= hpd68[1], 'accept_rate': acc_rate}

def write_mock_outputs(outdir_results: Path, floor: Optional[float]):
    rows = []
    plots_dir = outdir_results / 'plots'
    ensure_dirs(plots_dir)
    for mtrue in [0.06, 0.1]:
        res = mh_mock_recovery(mtrue, floor=floor, steps=30000, proposal_sigma=0.002, seed=42)
        rows.append(res)
        rng = np.random.default_rng(123)
        sigma_like = 0.0015

        def loglike(m):
            if floor is not None and m < floor:
                return -np.inf
            return -0.5 * ((m - mtrue) / sigma_like) ** 2
        m = max(mtrue, floor if floor is not None else -np.inf)
        chain = []
        for i in range(30000):
            prop = m + rng.normal(0, 0.002)
            a = loglike(prop) - loglike(m)
            if np.log(rng.random()) < a:
                m = prop
            chain.append(m)
        vals = np.array(chain)[3000:]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        plot_trace(axes[0], vals)
        axes[0].axhline(mtrue, color='k', ls='--', lw=1)
        plot_hist_kde(axes[1], vals)
        axes[1].axvline(mtrue, color='k', ls='--', lw=1, label='true')
        axes[1].legend()
        fig.suptitle(f'Mock recovery mnu_true={mtrue} eV')
        fig.tight_layout()
        fig.savefig(plots_dir / f'mock_recovery_{mtrue:.2f}.png', dpi=150)
        plt.close(fig)
    path = outdir_results / 'mock_recovery.csv'
    keys = ['mnu_true', 'mean', 'median', 'hpd68_lo', 'hpd68_hi', 'in_hpd68', 'accept_rate']
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_pdf_report(outdir_results: Path, summaries: List[Dict]):
    pdf_path = outdir_results / 'diagnostic_report.pdf'
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.1, 0.9, 'mnu Chain Diagnostics Report', fontsize=16, weight='bold')
        fig.text(0.1, 0.86, 'Getrimmte Posterioren sind bedingt durch prior/floor — enge Unsicherheiten können artefaktbedingt sein.', fontsize=9)
        fig.text(0.1, 0.83, 'Keine Detektionsaussage, bevor Mock-Recovery & Konvergenz-Checks abgeschlossen sind.', fontsize=9)
        fig.text(0.1, 0.79, f'Chains analyzed: {len(summaries)}', fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)
        for s in summaries:
            cname = s['chain_name']
            fig = plt.figure(figsize=(8.27, 11.69))
            y = 0.9

            def line(txt):
                nonlocal y
                fig.text(0.1, y, txt, fontsize=10)
                y -= 0.03
            line(f'Chain: {cname}')
            line(f"File: {s['file']}")
            line(f"N={s['N']}, mean={s['mean']:.6g}, std={s['std']:.3g}, median={s['p50']:.6g}")
            line(f"HPD68=[{s['hpd68_lo']:.6g}, {s['hpd68_hi']:.6g}], HPD95=[{s['hpd95_lo']:.6g}, {s['hpd95_hi']:.6g}]")
            line(f"P(mnu>0.06)={s['P_gt_0.06']:.3f}, P(mnu>0.1)={s['P_gt_0.1']:.3f}")
            line(f"tau_int≈{s['tau_int']:.2f}, N_eff≈{s['N_eff']:.0f}, R̂_split4={s['Rhat_split4']:.3f}")
            line(f"accept_rate_log: {(s['accept_rate_log'] if s['accept_rate_log'] is not None else 'NA')}")
            pdf.savefig(fig)
            plt.close(fig)
            plots = ['trace', 'hist', 'acf', 'cdf', 'compare_trimmed_untrimmed']
            for pl in plots:
                png = outdir_results / 'plots' / f'{cname}_{pl}.png'
                if png.exists():
                    img = plt.imread(str(png))
                    fig = plt.figure(figsize=(8.27, 5))
                    plt.imshow(img)
                    plt.axis('off')
                    pdf.savefig(fig)
                    plt.close(fig)

def write_next_steps(outdir_analysis: Path):
    p = outdir_analysis / 'next_steps.md'
    text = '# Next steps (priorisiert)\n\n1) Reparametrisierung und 4 Chains neu starten: y = log(mnu - 0.059 + 1e-6); proposal_scale ~ 0.5; learn_proposal: true; Starts mnu=[0.005, 0.06, 0.1, 0.2].\n\n2) Mock-Recovery (0.06, 0.10 eV) laufen lassen und Recovery prüfen (Median, 68% HPD, true ∈ HPD?).\n\n3) Falls weiterhin R̂ − 1 > 0.01 oder N_eff < 2000: Samplerwechsel zu HMC/NUTS in Erwägung ziehen.\n\n4) Dokumentation der Prior-Bedingung (floor) in den Resultaten explizit machen – getrimmte Posterioren sind bedingt.\n\nHinweis: Keine automatischen Änderungen an Produktions-Konfigurationen; bitte alle Vorschläge manuell prüfen (kein Auto-Merge).\n'
    with p.open('w', encoding='utf-8') as f:
        f.write(text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chain-file', action='append', required=True, help='Path to Cobaya chain .txt (can pass multiple)')
    ap.add_argument('--burnin_frac', type=float, default=0.1)
    ap.add_argument('--trim_floor', type=float, default=0.059, help='For overlay and trimmed stats; set None to disable')
    args = ap.parse_args()
    chain_files = [Path(p) for p in args.chain_file]
    root = Path(__file__).resolve().parents[1]
    out_results = root / 'results'
    out_plots = out_results / 'plots'
    out_analysis = root / 'analysis' / 'zencoder_output'
    ensure_dirs(out_results, out_plots, out_analysis)
    summaries = []
    for cf in chain_files:
        if not cf.exists():
            print(f'WARN: chain not found: {cf}')
            continue
        s = analyze_chain(cf, out_results, out_plots, args.burnin_frac, args.trim_floor)
        summaries.append(s)
    if not summaries:
        print('No chains analyzed. Exiting.')
        return 1
    write_summary_csv_md(summaries, out_results)
    write_diagnostics_csv(summaries, out_results)
    write_pdf_report(out_results, summaries)
    write_sample_requirements(summaries, out_results)
    build_suggested_yaml(summaries, out_analysis)
    write_mock_outputs(out_results, floor=args.trim_floor)
    write_next_steps(out_analysis)
    print('Done. Artifacts in results/ and analysis/zencoder_output/.')
    return 0
# kapitel: main
if __name__ == '__main__':
    sys.exit(main())

# kapitel: ende

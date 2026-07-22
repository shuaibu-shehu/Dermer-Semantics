"""Regenerate the paper figures from the experiment results.

Run from anywhere:  python "papers/ML4H 2026/make_figures.py"
Reads results/nbC2_fitz_results.json; writes figures/{audit,faithfulness,mitigation}.pdf
next to this script. Requires numpy + matplotlib.
"""
import json, os
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed TrueType, avoid Type-3 (venue-safe)
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 200,
})
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)
cb_res = json.load(open(os.path.join(REPO, 'results', 'nbC2_fitz_results.json'), encoding='utf-8'))
zs = cb_res['zero_shot_audit']; cb = cb_res['concept_bottleneck']

GROUPS = ['light', 'mid', 'dark']
LABELS = ['I-II\n(light)', 'III-IV\n(mid)', 'V-VI\n(dark)']
COL = {'light': '#4C72B0', 'mid': '#DD8452', 'dark': '#C44E52'}
x = np.arange(3)

def err(d, key='auc', lo='ci_lo', hi='ci_hi'):
    m = np.array([d[g][key] for g in GROUPS])
    l = np.array([d[g][key] - d[g][lo] for g in GROUPS])
    u = np.array([d[g][hi] - d[g][key] for g in GROUPS])
    return m, np.vstack([l, u])

# Figure 1: equity audit (2 panels)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
panels = [(axes[0], cb['fitz_test']['erm']['per_group'], 'Fitzpatrick17k (concept bottleneck)'),
          (axes[1], zs['ddi_external'], 'DDI, biopsy-proven (zero-shot)')]
for ax, d, title in panels:
    m, e = err(d)
    for i, g in enumerate(GROUPS):
        ax.errorbar(i, m[i], yerr=e[:, i:i+1], fmt='o', color=COL[g], ms=6,
                    capsize=4, elinewidth=1.4, mec='black', mew=0.5)
    ax.axhline(0.5, ls=':', c='0.5', lw=1)
    # label the chance line on the left, where no error bar reaches down to 0.5
    # (the right side collided with the dark-group CI in the DDI panel)
    ax.text(0.35, 0.508, 'chance', c='0.5', fontsize=7, va='bottom', ha='left')
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(0.42, 0.92); ax.set_title(title, fontsize=8.5)
axes[0].set_ylabel('Malignancy AUROC')
axes[0].annotate('gap 0.162\n$p<10^{-4}$', xy=(2, 0.679), xytext=(1.05, 0.60), fontsize=7.5,
                 ha='left', arrowprops=dict(arrowstyle='->', color='0.35', lw=0.9))
axes[1].annotate('disjoint\n95% CIs', xy=(2, 0.564), xytext=(0.9, 0.60), fontsize=7.5,
                 ha='left', arrowprops=dict(arrowstyle='->', color='0.35', lw=0.9))
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'audit.pdf'), bbox_inches='tight'); plt.close(fig)

# Figure 2: faithfulness (monotonicity) by tone
fig, ax = plt.subplots(figsize=(3.3, 2.7))
off = {'light': -0.18, 'mid': 0.0, 'dark': 0.18}
for tag, base in [('fitz_test', 0.0), ('ddi_external', 1.0)]:
    pg = cb[tag]['erm']['per_group']
    for g in GROUPS:
        d = pg[g]
        ax.errorbar(base + off[g], d['mono'],
                    yerr=[[d['mono'] - d['mono_lo']], [d['mono_hi'] - d['mono']]],
                    fmt='o', color=COL[g], ms=6, capsize=4, elinewidth=1.4, mec='black',
                    mew=0.5, label=g if base == 0.0 else None)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Fitzpatrick17k\n(disjoint CIs)', 'DDI\n(overlapping)'])
ax.set_ylabel('Explanation faithfulness\n(monotonicity)')
ax.set_xlim(-0.5, 1.5); ax.set_ylim(0.56, 0.70)
ax.legend(title='skin tone', frameon=False, loc='lower left', handletextpad=0.2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'faithfulness.pdf'), bbox_inches='tight'); plt.close(fig)

# Figure 3: Group-DRO does not close the gap
fig, ax = plt.subplots(figsize=(3.3, 2.7))
pe = cb['fitz_test']['erm']['per_group']; pg = cb['fitz_test']['gdro']['per_group']
w = 0.36
for i, g in enumerate(GROUPS):
    e_e = [[pe[g]['auc'] - pe[g]['ci_lo']], [pe[g]['ci_hi'] - pe[g]['auc']]]
    e_g = [[pg[g]['auc'] - pg[g]['ci_lo']], [pg[g]['ci_hi'] - pg[g]['auc']]]
    ax.bar(i - w/2, pe[g]['auc'], w, yerr=e_e, capsize=3, color='#B0B0B0', ec='black', lw=0.5,
           label='ERM' if i == 0 else None, error_kw={'elinewidth': 1})
    ax.bar(i + w/2, pg[g]['auc'], w, yerr=e_g, capsize=3, color=COL[g], ec='black', lw=0.5,
           label='Group-DRO' if i == 0 else None, error_kw={'elinewidth': 1})
ax.set_xticks(x); ax.set_xticklabels(LABELS)
ax.set_ylabel('Malignancy AUROC (Fitzpatrick17k)'); ax.set_ylim(0.5, 0.95)
ax.set_title('Group-DRO does not close the gap', fontsize=8.5)
ax.legend(frameon=False, loc='upper right', handletextpad=0.4)
# worst-group Delta = +0.014 [-0.011, 0.041], n.s. -- stated in the caption, kept off the bars
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'mitigation.pdf'), bbox_inches='tight'); plt.close(fig)

print('wrote figures to', FIG, ':', sorted(os.listdir(FIG)))

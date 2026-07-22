"""Diagnostic figure: where the skin-tone disparity lives.

Two panels, both computed on CPU from the cached DermLIP features (no Kaggle, no GPU):
  (a) BACKBONE side -- per-concept malignancy AUROC of DermLIP's zero-shot concept
      score, within each skin-tone group. Shows whether the concepts themselves
      carry less signal on darker skin.
  (b) HEAD side -- per-concept faithfulness *violation* rate, within each tone group:
      the fraction of lesions for which forcing a melanoma criterion ON *lowers*
      P(malignant). Decomposes the aggregate monotonicity number in the paper into
      which of the 7 concepts break, and for whom.

Run:  python "papers/ML4H 2026/make_diagnostic_figures.py"
Reads caches/fairness_cache/features_{fitz,ddi}.npz ; writes figures/diagnostic.pdf .
Requires numpy + torch + matplotlib. Training matches notebook 04 exactly
(HID=16, iters=1500, lr=0.05, seeds 0/1/2, pos-weighted BCE) so the per-concept
numbers decompose the same ERM ensemble the main results report.
"""
import os, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 200,
})
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
FIG = os.path.join(HERE, 'figures')
CACHE = os.path.join(REPO, 'caches', 'fairness_cache')
os.makedirs(FIG, exist_ok=True)

GROUPS = ['light', 'mid', 'dark']
COL = {'light': '#4C72B0', 'mid': '#DD8452', 'dark': '#C44E52'}
# order MUST match CONCEPT_PROMPTS in notebook 03
CONCEPTS = ['atypical_pigment_network', 'blue_whitish_veil', 'atypical_vascular',
            'irregular_streaks', 'irregular_pigmentation', 'irregular_dots_globules', 'regression']
NICE = ['atypical pigment\nnetwork', 'blue-whitish\nveil', 'atypical\nvascular',
        'irregular\nstreaks', 'irregular\npigmentation', 'irregular dots\n/ globules', 'regression\nstructures']

# ------------------------------------------------------------------ load cache
def load_tag(tag):
    z = np.load(os.path.join(CACHE, 'features_%s.npz' % tag), allow_pickle=True)
    C = z['concept_scores'].astype(np.float32)
    g = z['group'].astype('U8')
    ok = np.isfinite(C).all(1) & np.isin(g, GROUPS)
    return {'C': torch.tensor(C[ok]), 'y': torch.tensor(z['malignant'][ok].astype(np.float32)),
            'g': g[ok], 'split': z['split'].astype('U8')[ok]}

fitz, ddi = load_tag('fitz'), load_tag('ddi')
tr = fitz['split'] == 'train'
te = fitz['split'] == 'test'
Xtr, ytr, gtr = fitz['C'][tr], fitz['y'][tr], fitz['g'][tr]

# ------------------------------------------------------------------ model (== nb04)
def fwd(C, P):
    W1, b1, W2, b2 = P
    return (torch.relu(C @ W1 + b1) @ W2 + b2).squeeze(1)

def auc(y_true, score):
    y_true = np.asarray(y_true).astype(float); score = np.asarray(score).astype(float)
    pos = score[y_true == 1]; neg = score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg]); rank = allv.argsort().argsort().astype(float) + 1
    return (rank[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

HID = 16
def init(seed):
    gg = torch.Generator().manual_seed(seed)
    W1 = (torch.randn(Xtr.shape[1], HID, generator=gg) * 0.3).requires_grad_(True)
    b1 = torch.zeros(HID, requires_grad=True)
    W2 = (torch.randn(HID, 1, generator=gg) * 0.3).requires_grad_(True)
    b2 = torch.zeros(1, requires_grad=True)
    return [W1, b1, W2, b2]

def train_erm(seed, iters=1500, lr=0.05):
    P = init(seed); opt = torch.optim.Adam(P, lr=lr)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    pw = torch.tensor(max(float((ytr == 0).sum()), 1.0) / max(float((ytr == 1).sum()), 1.0))
    for _ in range(iters):
        loss = bce(fwd(Xtr, P), ytr, pos_weight=pw, reduction='mean')
        opt.zero_grad(); loss.backward(); opt.step()
    return P

SEEDS = [0, 1, 2]
heads = [train_erm(s) for s in SEEDS]

# ------------------------------------------------------------------ (a) concept separability by tone (backbone)
def concept_auc(data, j, grp):
    m = (data['g'] == grp)
    return auc(data['y'].numpy()[m], data['C'].numpy()[m, j])

sep = {g: [concept_auc(fitz, j, g) for j in range(7)] for g in GROUPS}   # fitz TEST subset below
# restrict to test split for the in-distribution story (matches paper eval)
fitz_te = {'C': fitz['C'][te], 'y': fitz['y'][te], 'g': fitz['g'][te]}
sep = {g: [concept_auc(fitz_te, j, g) for j in range(7)] for g in GROUPS}

# ------------------------------------------------------------------ (b) per-concept faithfulness violation by tone (head)
def viol_per_concept(data, grp):
    """violation rate for each concept within group grp, averaged over the ERM heads."""
    m = (data['g'] == grp)
    C = data['C'][m]
    out = []
    for j in range(7):
        v1 = C.clone(); v1[:, j] = 1.0
        v0 = C.clone(); v0[:, j] = 0.0
        vs = []
        for P in heads:
            p1 = torch.sigmoid(fwd(v1, P)); p0 = torch.sigmoid(fwd(v0, P))
            vs.append((p1 < p0 - 1e-6).float().mean().item())   # forcing ON lowers P => violation
        out.append(float(np.mean(vs)))
    return out

viol = {g: viol_per_concept(fitz_te, g) for g in GROUPS}

# save the numbers next to the results so the paper text can cite them
diag = {'concept_order': CONCEPTS,
        'separability_auc_fitz_test': sep,
        'violation_rate_fitz_test': viol}
with open(os.path.join(REPO, 'results', 'nbC2_fitz_diagnostic.json'), 'w') as fh:
    json.dump(diag, fh, indent=2)

# ------------------------------------------------------------------ figure
fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.8, 3.2))
y = np.arange(7)[::-1]        # top-to-bottom = concept order
h = 0.24

# (a) separability AUROC
for k, g in enumerate(GROUPS):
    axL.barh(y + (1 - k) * h, sep[g], height=h, color=COL[g], ec='black', lw=0.4, label=g)
axL.axvline(0.5, ls=':', c='0.5', lw=1)
# label the chance line in the empty gap beside the (sub-chance) vascular row,
# not at the top where it collided with the title
axL.text(0.53, 4.0, 'chance', c='0.55', fontsize=6.5, ha='left', va='center')
axL.set_yticks(y); axL.set_yticklabels(NICE)
axL.set_xlim(0.3, 0.9); axL.set_ylim(-0.6, 6.6); axL.set_xlabel('per-concept malignancy AUROC')
axL.set_title('(a) do the concepts carry signal?\n(DermLIP backbone)', fontsize=8.5)
axL.legend(title='skin tone', frameon=False, loc='lower right', handletextpad=0.4, fontsize=7.5)

# (b) violation rate
for k, g in enumerate(GROUPS):
    axR.barh(y + (1 - k) * h, viol[g], height=h, color=COL[g], ec='black', lw=0.4)
axR.set_yticks(y); axR.set_yticklabels([]); axR.set_ylim(-0.6, 6.6)
axR.set_xlim(0, max(0.5, max(max(v) for v in viol.values()) * 1.15))
axR.set_xlabel('faithfulness violation rate\n(forcing criterion ON lowers risk)')
axR.set_title('(b) does the head honour them?\n(trained bottleneck)', fontsize=8.5)

fig.tight_layout()
fig.savefig(os.path.join(FIG, 'diagnostic.pdf'), bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------ console summary
print('wrote', os.path.join(FIG, 'diagnostic.pdf'))
print('\nconcept separability AUROC (fitz test), light / dark:')
for j, c in enumerate(CONCEPTS):
    print('  %-26s light=%.3f mid=%.3f dark=%.3f' % (c, sep['light'][j], sep['mid'][j], sep['dark'][j]))
print('\nfaithfulness violation rate (fitz test), light / dark:')
for j, c in enumerate(CONCEPTS):
    print('  %-26s light=%.3f mid=%.3f dark=%.3f' % (c, viol['light'][j], viol['mid'][j], viol['dark'][j]))
mean_sep = {g: float(np.mean(sep[g])) for g in GROUPS}
mean_vio = {g: float(np.mean(viol[g])) for g in GROUPS}
print('\nmean over concepts | separability', {k: round(v, 3) for k, v in mean_sep.items()},
      '| violation', {k: round(v, 3) for k, v in mean_vio.items()})

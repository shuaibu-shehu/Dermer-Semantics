"""Figure 1: a polished, correct, fully-vector schematic of the pipeline.

No data, no GPU, no embedded photo -- everything (lesion icon, concept sliders,
MLP head, risk gauge) is drawn in matplotlib, so it stays sharp at any size and
carries no image-licensing baggage. Laid out with a compact aspect ratio so it
does not eat vertical space on the page.

Correct by construction (unlike an image-generator mockup):
  * DermLIP IS the frozen encoder (one stage, labelled as such).
  * exactly 7 concept sliders, scored absent -> present.
  * the output is P(malignant), a probability on a 0-1 dial -- NOT "present/absent".
  * 3 skin-tone groups (light/mid/dark), matching the audit in the paper.

Run:  python "papers/ML4H 2026/make_pipeline_figure.py"  ->  figures/pipeline.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 9})
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

NEUT_F, NEUT_E = '#EFEFEF', '#7A7A7A'
FROZ_F, FROZ_E = '#DCE7F2', '#3B6EA5'
CONC_F, CONC_E = '#E4EEE4', '#5B8A5B'
HEAD_F, HEAD_E = '#FBE3CF', '#C87A34'
OUT_F,  OUT_E  = '#F5D9DA', '#C44E52'
PURPLE = '#7A3B9A'
TONE = {'light': '#4C72B0', 'mid': '#DD8452', 'dark': '#C44E52'}

fig, ax = plt.subplots(figsize=(7.4, 1.96))
ax.set_xlim(0, 100); ax.set_ylim(12.5, 39); ax.axis('off')

def rbox(x, y, w, h, fc, ec, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6,rounding_size=2',
                                fc=fc, ec=ec, lw=lw, zorder=1))

def arrow(x1, y1, x2, y2, c='#444', lw=1.5, ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12,
                                 lw=lw, color=c, linestyle=ls, shrinkA=0, shrinkB=0, zorder=3))

YB, HB = 22, 14          # box bottom / height (compact)
YM = YB + HB / 2         # box vertical middle (=29)

B1 = (1, 15); B2 = (20, 20); B3 = (45, 19); B4 = (69, 16); B5 = (89, 10)
rbox(B1[0], YB, B1[1], HB, NEUT_F, NEUT_E)
rbox(B2[0], YB, B2[1], HB, FROZ_F, FROZ_E)
rbox(B3[0], YB, B3[1], HB, CONC_F, CONC_E)
rbox(B4[0], YB, B4[1], HB, HEAD_F, HEAD_E)
rbox(B5[0], YB, B5[1], HB, OUT_F, OUT_E)

# ---- (1) stylised lesion icon ------------------------------------------------
lx, ly = B1[0] + B1[1] / 2, YB + HB * 0.62
ax.add_patch(Circle((lx, ly), 3.9, fc='#E8C6A0', ec='#C79A6B', lw=1.0, zorder=2))
th = np.linspace(0, 2 * np.pi, 60)
rng = np.random.default_rng(4)
rr = 1 + 0.16 * np.sin(3 * th + 0.6) + 0.10 * np.sin(5 * th + 1.2) + 0.05 * rng.standard_normal(60)
ax.add_patch(Polygon(np.c_[lx + 2.7 * rr * np.cos(th), ly + 2.7 * rr * np.sin(th)],
                     closed=True, fc='#5A3A2E', ec='#3E2A22', lw=0.8, zorder=2))
ax.add_patch(Circle((lx - 0.9, ly + 0.6), 0.8, fc='#3E2A22', ec='none', zorder=2))
ax.add_patch(Circle((lx + 1.0, ly - 0.8), 0.6, fc='#734A38', ec='none', zorder=2))
ax.text(lx, YB + 2.1, 'skin lesion\nimage', ha='center', va='center', fontsize=7.3, zorder=2)

# ---- (2) DermLIP frozen encoder ----------------------------------------------
cx2 = B2[0] + B2[1] / 2
ax.text(cx2, YB + HB - 3.0, 'DermLIP', ha='center', va='center', fontsize=10,
        fontweight='bold', color=FROZ_E, zorder=2)
ax.text(cx2, YM - 0.2, '❄', ha='center', va='center', fontsize=15, color=FROZ_E, zorder=2)
ax.text(cx2, YB + 2.4, 'frozen encoder', ha='center', va='center', fontsize=7.8, zorder=2)
ax.text(cx2, YB + HB + 1.4, 'frozen', ha='center', va='center', fontsize=7, color=FROZ_E)

# ---- (3) seven concept-score sliders -----------------------------------------
cx3 = B3[0] + B3[1] / 2
ax.text(cx3, YB + HB - 2.2, '7 concept scores', ha='center', va='center',
        fontsize=8.3, fontweight='bold', color=CONC_E, zorder=2)
ax.text(cx3, YB + HB - 4.5, '$c \\in \\mathbb{R}^{7}$', ha='center', va='center',
        fontsize=7.2, color=CONC_E, zorder=2)
s_x0, s_x1 = B3[0] + 2.8, B3[0] + B3[1] - 2.8
vals = [0.72, 0.55, 0.30, 0.82, 0.46, 0.63, 0.38]
s_ys = np.linspace(YB + HB - 6.0, YB + 2.9, 7)
for v, sy in zip(vals, s_ys):
    ax.plot([s_x0, s_x1], [sy, sy], '-', color='#BBBBBB', lw=1.0, zorder=2, solid_capstyle='round')
    ax.plot([s_x0 + v * (s_x1 - s_x0)], [sy], 'o', ms=3.4, color=CONC_E, mec='white', mew=0.5, zorder=3)
ax.text(s_x0, YB + 1.3, 'absent', ha='left', va='center', fontsize=5.4, color='#8a8a8a')
ax.text(s_x1, YB + 1.3, 'present', ha='right', va='center', fontsize=5.4, color='#8a8a8a')

# ---- (4) small trained head (tiny MLP) ---------------------------------------
cx4 = B4[0] + B4[1] / 2
ax.text(cx4, YB + HB - 2.7, 'small head', ha='center', va='center', fontsize=8.7,
        fontweight='bold', color=HEAD_E, zorder=2)
ax.text(cx4, YB + HB - 5.0, '$g_\\theta$', ha='center', va='center', fontsize=8.5, zorder=2)
ax.text(cx4, YB + HB + 1.4, 'trained', ha='center', va='center', fontsize=7, color=HEAD_E)
cols = [B4[0] + 4.5, cx4, B4[0] + B4[1] - 4.5]
layer_y = [np.linspace(YB + 2.4, YB + 6.8, 3), np.linspace(YB + 3.0, YB + 6.2, 2), [YB + 4.6]]
for a in range(2):
    for ya in layer_y[a]:
        for yb in layer_y[a + 1]:
            ax.plot([cols[a], cols[a + 1]], [ya, yb], '-', color='#E3B588', lw=0.6, zorder=2)
for cxn, ys in zip(cols, layer_y):
    for yn in ys:
        ax.add_patch(Circle((cxn, yn), 0.72, fc=HEAD_E, ec='white', lw=0.5, zorder=3))

# ---- (5) P(malignant) as a 0-1 risk dial -------------------------------------
cx5, gy = B5[0] + B5[1] / 2, YB + 4.0
ax.text(cx5, YB + HB - 2.7, '$P(\\mathrm{mal})$', ha='center', va='center', fontsize=9,
        fontweight='bold', color=OUT_E, zorder=2)
r_g = 2.9
ax.add_patch(Wedge((cx5, gy), r_g, 0, 180, width=0.9, fc='#EBB9BB', ec=OUT_E, lw=0.6, zorder=2))
ang = np.deg2rad(180 * (1 - 0.72))
ax.plot([cx5, cx5 + 0.8 * r_g * np.cos(ang)], [gy, gy + 0.8 * r_g * np.sin(ang)],
        '-', color=OUT_E, lw=1.5, zorder=3, solid_capstyle='round')
ax.add_patch(Circle((cx5, gy), 0.4, fc=OUT_E, ec='none', zorder=3))
ax.text(cx5 - r_g, gy - 1.2, '0', ha='center', fontsize=6, color='#777')
ax.text(cx5 + r_g, gy - 1.2, '1', ha='center', fontsize=6, color='#777')

# ---- arrows (stop short of the rounded borders) ------------------------------
def rt(b):
    return b[0] + b[1]
for a, b in [(B1, B2), (B2, B3), (B3, B4), (B4, B5)]:
    arrow(rt(a) + 0.7, YM, b[0] - 1.0, YM)

# ---- annotations below (pulled in close) -------------------------------------
ax.text((rt(B2) + B3[0]) / 2, YB - 1.8, 'present / absent\ntext prompts',
        ha='center', va='top', fontsize=6.4, color='#666', style='italic')

arrow(cx3, YB - 0.3, cx3, YB - 3.6, c=PURPLE, lw=1.1, ls=(0, (3, 2)))
ax.text(cx3, YB - 4.4, 'faithfulness test', ha='center', va='top', fontsize=7.4,
        fontweight='bold', color=PURPLE)
ax.text(cx3, YB - 6.5, r'force a criterion on ($c_i\!\rightarrow\!1$) $\Rightarrow$ $P(\mathrm{mal})$ must not drop',
        ha='center', va='top', fontsize=6.7, color=PURPLE)

arrow(cx5, YB - 0.3, cx5, YB - 3.0, c='#555', lw=1.0, ls=(0, (3, 2)))
ax.text(cx5, YB - 3.8, 'audit by\nskin tone', ha='center', va='top', fontsize=6.6, color='#333')
for i, g in enumerate(['light', 'mid', 'dark']):
    ax.plot(cx5 + (i - 1) * 2.3, YB - 7.4, 'o', ms=4.6, color=TONE[g], mec='black', mew=0.4)

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(os.path.join(FIG, 'pipeline.pdf'), bbox_inches='tight', pad_inches=0.02)
fig.savefig(os.path.join(HERE, '_pipeline_preview.png'), bbox_inches='tight', dpi=170, pad_inches=0.02)
plt.close(fig)
print('wrote', os.path.join(FIG, 'pipeline.pdf'))

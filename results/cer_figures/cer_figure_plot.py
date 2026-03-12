# Python
import matplotlib.pyplot as plt
import numpy as np

UCLA_COLORS = ['#2D68C4', '#FFD100', '#003B5C', '#8BB8E8', '#00A5E0', '#005587']

baseline_color = '#FFD100'
baseline_full_cer = 22.28 # [%]
baseline_latent_cer = 21.0 # [%]
baseline_biophys_aug_cer = 24.45 # [%]

rnn_color = '#8BB8E8'
rnn_full_cer = 18.24 # [%]
rnn_latent_cer = 19.5 # [%]
rnn_biophys_aug_cer = 22.76 # [%]

hybrid_color =  '#00A5E0'
hybrid_full_cer = 14.98 # [%]
hybrid_latent_cer = 15.5 # [%]
hybrid_biophys_aug_cer = 16.76 # [%]

conformer_color = '#003B5C'
conformer_full_cer = 14.1 # [%]
conformer_latent_cer = 16.2 # [%]
conformer_biophys_aug_cer = 15.89 # [%]

# ---------------------------------------------------------------------------
# Bar chart — CER by model and data condition
# ---------------------------------------------------------------------------

models      = ['Baseline', 'RNN', 'Hybrid', 'Conformer']
colors      = [baseline_color, rnn_color, hybrid_color, conformer_color]
edge_colors = ['#B89800', '#5A88C8', '#007EAD', '#001F3D']   # darker outlines

full_cers        = [baseline_full_cer,        rnn_full_cer,        hybrid_full_cer,        conformer_full_cer]
latent_cers      = [baseline_latent_cer,      rnn_latent_cer,      hybrid_latent_cer,      conformer_latent_cer]
biophys_aug_cers = [baseline_biophys_aug_cer, rnn_biophys_aug_cer, hybrid_biophys_aug_cer, conformer_biophys_aug_cer]

n_models    = len(models)
n_bars      = 3           # conditions per group
group_width = 0.7         # total width occupied by bars in one group
bar_width   = group_width / n_bars
offsets     = np.array([-1, 0, 1]) * bar_width  # centre offsets for the 3 conditions
x           = np.arange(n_models)

condition_labels   = ['Full EMG', 'Latent AE', 'Biophys Aug']
condition_hatches  = ['', '///', '...']    # visual texture to aid B&W printing
condition_alphas   = [1.0, 0.82, 0.65]

# MAKE HATCHES BOLDER This must be set before the plot is created
plt.rcParams['hatch.linewidth'] = 2.0

plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.linewidth':   0.8,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'xtick.direction':  'out',
    'ytick.direction':  'out',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#AAAAAA',
    'legend.fontsize':  10,
    'pdf.fonttype':     42,    # embeds fonts for publication
    'ps.fonttype':      42,
})

fig, ax = plt.subplots(figsize=(8.5, 3), dpi=1200)

all_cers = [full_cers, latent_cers, biophys_aug_cers]

for cond_idx, (cers, label, hatch, alpha) in enumerate(
        zip(all_cers, condition_labels, condition_hatches, condition_alphas)):
    for m_idx, (cer, color, ec) in enumerate(zip(cers, colors, edge_colors)):
        xpos = x[m_idx] + offsets[cond_idx]
        bar = ax.bar(
            xpos, cer,
            width=bar_width * 0.92,
            color=color,
            edgecolor=ec,
            linewidth=0.7,
            hatch=hatch,
            alpha=alpha,
            zorder=3,
        )
        # Percentage label above bar
        # Manuall adjust spacings...
        xpos_for_label = xpos
        if cond_idx == 0:  # Full EMG
            xpos_for_label = xpos - 0.05 # shift left a little
        elif cond_idx == 1:  # Latent AE
            xpos_for_label = xpos - 0.04 # shift left a little
        elif cond_idx == 2:  # Biophys Aug
            xpos_for_label = xpos + 0.02 # shift right a little

        ax.text(
            xpos_for_label, cer + 0.25,
            f'{cer:.1f}%',
            ha='center', va='bottom',
            fontsize=12,
            fontweight='bold' if cond_idx == 0 else 'normal',
            color='#222222',
            rotation=0,
        )

# Legend: one patch per condition (hatches match)
from matplotlib.patches import Patch
legend_patches = [
    Patch(facecolor='#888888', edgecolor='#444444', hatch=h, alpha=a, label=lbl)
    for h, a, lbl in zip(condition_hatches, condition_alphas, condition_labels)
]
ax.legend(handles=legend_patches, loc='upper right', title='Data condition',
          title_fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel('Character Error Rate (%)', fontsize=12)
# ax.set_title('CER by Model Architecture and Data Condition',
#              fontsize=13, fontweight='bold', pad=12)

# Zoom in on the data range (starts at 12, gives 2% headroom above highest bar)
ax.set_ylim(12, 26)
# ax.set_ylim(0, max(biophys_aug_cers) + 5.5)

ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()

out_path = 'results/cer_figures/cer_comparison.png'
fig.savefig(out_path, dpi=1200, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.show()


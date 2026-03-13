import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Make hatches bolder
plt.rcParams['hatch.linewidth'] = 2.0

UCLA_COLORS = ['#2D68C4', '#FFD100', '#003B5C', '#8BB8E8', '#00A5E0', '#005587']

baseline_color = '#FFD100'
baseline_full_cer = 22.28
baseline_latent_cer = 85.9
baseline_biophys_aug_cer = 24.45

rnn_color = '#8BB8E8'
rnn_full_cer = 18.24
rnn_latent_cer = 104.03
rnn_biophys_aug_cer = 22.76

hybrid_color =  '#00A5E0'
hybrid_full_cer = 19.03
hybrid_latent_cer = 69.21
hybrid_biophys_aug_cer = 21.39

conformer_color = '#003B5C'
conformer_full_cer = 14.1
conformer_latent_cer = 84.95
conformer_biophys_aug_cer = 15.89

models      = ['Baseline', 'RNN', 'Hybrid', 'Conformer']
colors      = [baseline_color, rnn_color, hybrid_color, conformer_color]
edge_colors = ['#B89800', '#5A88C8', '#007EAD', '#001F3D']

full_cers        = [baseline_full_cer,        rnn_full_cer,        hybrid_full_cer,        conformer_full_cer]
latent_cers      = [baseline_latent_cer,      rnn_latent_cer,      hybrid_latent_cer,      conformer_latent_cer]
biophys_aug_cers = [baseline_biophys_aug_cer, rnn_biophys_aug_cer, hybrid_biophys_aug_cer, conformer_biophys_aug_cer]

n_models    = len(models)
n_bars      = 3
group_width = 0.7
bar_width   = group_width / n_bars
offsets     = np.array([-1, 0, 1]) * bar_width
x           = np.arange(n_models)

condition_labels   = ['Full EMG', 'Latent AE', 'Biophys Aug']
condition_hatches  = ['', '///', '...']
condition_alphas   = [1.0, 0.82, 0.65]

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
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
})

# --- DEFINE AXIS LIMITS FIRST ---
top_ylim = (60, 105)
bot_ylim = (13, 26)

# Provide manual height ratios to compress the top and expand the bottom
height_ratios = [1, 2]  # Top is much smaller than the bottom now

# --- BROKEN AXIS SETUP (2 subplots) ---
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(8.5, 3), dpi=1200, # slightly taller figure to accommodate the bottom
    gridspec_kw={'height_ratios': height_ratios}
)
# Narrower gap between the sections
fig.subplots_adjust(hspace=0.08)

all_cers = [full_cers, latent_cers, biophys_aug_cers]

# Lists to track which bars cross which breaks
top_break_x = []
actual_bar_width = bar_width * 0.92

# Plot the EXACT SAME data on all axes
for ax in [ax_top, ax_bot]:
    for cond_idx, (cers, label, hatch, alpha) in enumerate(
            zip(all_cers, condition_labels, condition_hatches, condition_alphas)):
        for m_idx, (cer, color, ec) in enumerate(zip(cers, colors, edge_colors)):
            xpos = x[m_idx] + offsets[cond_idx]
            bar = ax.bar(
                xpos, cer,
                width=actual_bar_width,
                color=color,
                edgecolor=ec,
                linewidth=0.7,
                hatch=hatch,
                alpha=alpha,
                zorder=3,
            )
            
            # --- CONDITIONAL TEXT LABELS ---
            xpos_for_label = xpos
            if cond_idx == 0: xpos_for_label = xpos - 0.05
            elif cond_idx == 1: xpos_for_label = xpos - 0.04
            elif cond_idx == 2: xpos_for_label = xpos + 0.04
                
            # Log the x-positions of the bars crossing the axes breaks (only needs to run once)
            if ax == ax_top:
                if cer > 30: top_break_x.append(xpos)
                
            # Only draw the label on the axis where the top of the bar actually ends
            if ax == ax_top and cer > 60:
                # Target the tallest bar specifically (RNN Model is m_idx == 1, Latent AE is cond_idx == 1)
                if m_idx == 1 and cond_idx == 1:
                    ax.text(xpos_for_label-0.28, cer - 10, f'{cer:.1f}%', ha='center', va='bottom',
                            fontsize=10, fontweight='bold' if cond_idx == 0 else 'normal', color='#222222')
                else:
                    # The other top bars stay centered above the bar
                    # Just fully centered since its top, so xpos
                    ax.text(xpos, cer + 2, f'{cer:.1f}%', ha='center', va='bottom',
                            fontsize=10, fontweight='bold' if cond_idx == 0 else 'normal', color='#222222')
            elif ax == ax_bot and 10 < cer < 30:
                ax.text(xpos_for_label, cer + 0.35, f'{cer:.1f}%', ha='center', va='bottom',
                        fontsize=11, fontweight='bold' if cond_idx == 0 else 'normal', color='#222222')

# --- AXIS LIMITS & SPINES ---
ax_top.set_ylim(*top_ylim)
ax_bot.set_ylim(*bot_ylim)

# Hide inner spines where the breaks happen
ax_top.spines.bottom.set_visible(False)
ax_bot.spines.top.set_visible(False)

# Hide x ticks and labels on top plot so they only show at the bottom
ax_top.tick_params(labelbottom=False, bottom=False)  

for ax in [ax_top, ax_bot]:
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

# Fetch data coordinates of the y-axis spine for the breaks
# CRITICAL: Freeze the x-limits so drawing the wave doesn't auto-expand the axis
xlim = ax_top.get_xlim()
for ax in [ax_top, ax_bot]:
    ax.set_xlim(xlim)
    
spine_x = xlim[0]

# --- LOCALIZED WAVY BREAK LOGIC ---
def draw_local_breaks(ax_upper, ax_lower, x_centers, width, amp_up=0.03, amp_down=0.03, freq=2.0):
    """Draws a sine wave ONLY over specific x intervals (bars or spines)."""
    trans_upper = ax_upper.get_xaxis_transform() # x in data coords, y in axes coords
    trans_lower = ax_lower.get_xaxis_transform()
    
    for xc in x_centers:
        x_line = np.linspace(xc - width/2, xc + width/2, 100)
        
        # Upper wave (masks the straight bottom edge of the upper axis)
        y_wave_up = amp_up + amp_up * np.sin(2 * np.pi * freq * (x_line - x_line[0]) / width)
        ax_upper.fill_between(x_line, 0, y_wave_up, facecolor='white', edgecolor='none', transform=trans_upper, zorder=4, clip_on=False)
        ax_upper.plot(x_line, y_wave_up, color='k', linewidth=0.5, linestyle=':', transform=trans_upper, zorder=5, clip_on=False)
        
        # Lower wave (masks the straight top edge of the lower axis)
        y_wave_down = 1 - amp_down + amp_down * np.sin(2 * np.pi * freq * (x_line - x_line[0]) / width)
        ax_lower.fill_between(x_line, y_wave_down, 1, facecolor='white', edgecolor='none', transform=trans_lower, zorder=4, clip_on=False)
        ax_lower.plot(x_line, y_wave_down, color='k', linewidth=0.5, linestyle=':', transform=trans_lower, zorder=5, clip_on=False)

# Auto-calculate wave amplitudes based on the dynamic height ratios
top_ratio = height_ratios[0] / sum(height_ratios)
bot_ratio = height_ratios[1] / sum(height_ratios)

base_amp = 0.03

# Break between Top and Bot subplots
# Because the ratios are now heavily skewed, we need to balance the visual amplitudes carefully.
# If top is squished, its wave needs a smaller amplitude in axes coords to look 'normal'.
amp_top_down = base_amp * (bot_ratio / top_ratio) * 0.5 # Add a dampener because the skew is so large
amp_bot_up = base_amp * 0.8 

draw_local_breaks(ax_top, ax_bot, top_break_x, actual_bar_width, amp_up=amp_top_down, amp_down=amp_bot_up, freq=2.0)
draw_local_breaks(ax_top, ax_bot, [spine_x], 0.075, amp_up=amp_top_down, amp_down=amp_bot_up, freq=1.0) 

# --- LEGEND & LABELS ---
legend_patches = [Patch(facecolor='#888888', edgecolor='#444444', hatch=h, alpha=a, label=lbl)
                  for h, a, lbl in zip(condition_hatches, condition_alphas, condition_labels)]
ax_top.legend(handles=legend_patches, loc='upper left', title='Data condition',
          title_fontsize=10, bbox_to_anchor=(1.02, 1)) 

ax_bot.set_xticks(x)
ax_bot.set_xticklabels(models, fontsize=12)

# Set the y-axis label to straddle both plots nicely
fig.supylabel('Character Error Rate (%)', fontsize=12, x=0.04)

out_path = 'results/cer_figures/cer_comparison.png' # Github path
# out_path = 'cer_comparison.png' # Colab testing path
fig.savefig(out_path, dpi=1200, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.show()
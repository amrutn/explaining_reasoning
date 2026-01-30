import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D

def calculate_diff_grid(n_grid, m_grid, d, proportionality_const=10**(-12)):
    alpha = 2 / d
    term_val = m_grid ** alpha
    prod_val = term_val ** n_grid
    sum_val = n_grid * term_val
    return (prod_val - sum_val) * proportionality_const

# Grid setup
n_vals = np.linspace(2, 25, 1000)
m_vals = np.linspace(1.1, 5, 1000) 
M, N = np.meshgrid(m_vals, n_vals)

d = 3
Z = calculate_diff_grid(N, M, d)

plt.figure(figsize=(4, 3))
plt.rcParams.update({
    'font.size': 14,        # Base font size
    'axes.labelsize': 14,   # Axis labels
    'xtick.labelsize': 12,  # Tick labels
    'ytick.labelsize': 12,
})

# SymLogNorm
norm = colors.SymLogNorm(linthresh=10**(-12), linscale=1.0, vmin=Z.min(), vmax=Z.max(), base=10)

mesh = plt.pcolormesh(M, N, Z, cmap='RdBu_r', norm=norm, shading='auto', rasterized=True)

# Add lines (isoclines for constant N roughly)
for i in range(10):
    val = float(10**(2*i+1))
    # Avoid division by zero or log of negative numbers if m_vals <= 1
    # m_vals are linspace(1.1, 5) so we are safe
    plt.plot(m_vals, np.log(val)/np.log(m_vals), 'k-', alpha=.3)

plt.ylim(2,25)

# Vertical line at optimal m
plt.axvline(np.exp(d/2), c='k', ls=':', alpha=.6)

# 1. Draw the arrow separately (Empty text string)
#    Moved xytext (tail) down to (2.4, 10) from the original (2.5, 16)
plt.annotate('', 
             xy=(3.8, 22),         # Arrow tip
             xytext=(2.4, 10),     # Arrow tail (start location)
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

# 2. Place the text separately (Offset above the arrow)
plt.text(2.2, 10.5, r'Increasing $N$', 
         fontsize=10, 
         color='black',
         rotation=55,              # Aligned with the arrow's slope
         ha='left', va='bottom')

# Custom Ticks
max_exp = int(np.ceil(np.log10(Z.max())))
exponents = range(-11, max_exp + 1, 4)
pos_ticks = [10**e for e in exponents]
neg_ticks = [-10**e for e in exponents if -10**e >= Z.min()]
custom_ticks = sorted(neg_ticks + [0] + pos_ticks)

# Colorbar with rectangular extensions
cbar = plt.colorbar(mesh, extend='both', extendrect=True, ticks=custom_ticks)
cbar.set_label(r'Reasoning Gain', rotation=270, labelpad=5, fontsize=12)

# Black dashed contour for Z=0
plt.contour(M, N, Z, levels=[0], colors='k', linestyles='--')

plt.xlabel('Degree $m$', fontsize=12)
plt.ylabel('Depth $n$', fontsize=12)

# --- Add Compact Legend ---
h_opt = Line2D([0], [0], color='k', linestyle=':', linewidth=1.5, alpha=0.6)
h_nodiff = Line2D([0], [0], color='k', linestyle='--', linewidth=1.5)
h_const = Line2D([0], [0], color='k', linestyle='-', alpha=0.3)

plt.xticks([2,3,np.exp(d/2)], ['2','3',r'$m^*$'])
plt.tight_layout()
plt.savefig('heatmap_reasoning_theory.pdf', dpi=100, bbox_inches='tight')
plt.show()
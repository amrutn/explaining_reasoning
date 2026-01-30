import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator

# Update font sizes globally
plt.rcParams.update({
    'font.size': 14,        # Base font size
    'axes.labelsize': 14,   # Axis labels
    'xtick.labelsize': 12,  # Tick labels
    'ytick.labelsize': 12,
})

d = 2

# m from 1 to 10
m_vals = np.logspace(0, 1, 500) 
# r from 1 to 5
r_vals = np.linspace(1, 5, 500)
# c*D^{-1/d}
proportionality_const = 0.01

M, R = np.meshgrid(m_vals, r_vals)

X = M ** (2/d)
Z = (X - R * (X ** (1/R)))*proportionality_const

plt.figure(figsize=(4, 3))

vmin = Z.min()
vmax = Z.max()
divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)

pcm = plt.pcolormesh(M, R, Z, cmap='RdBu', shading='auto', norm=divnorm, rasterized=True)

# Colorbar with increased font size for label and ticks
cbar = plt.colorbar(pcm)
# Using \mathrm instead of \text for robustness in matplotlib's internal parser
# Visually identical (upright font in math mode)
cbar.set_label(r'Thinking Gain', fontsize=14, rotation=270, labelpad=15)
cbar.ax.tick_params(labelsize=12)
cbar.locator = MaxNLocator(nbins=5) 
cbar.update_ticks()

plt.contour(M, R, Z, levels=[0], colors='black', linewidths=2, linestyles='dashed')
plt.xscale('log')
plt.xticks([1,np.exp(d/2),10], [r'$10^0$',r'$m^*$',r'$10^1$'])
plt.xlabel(r'Degree $m$', fontsize=14)
plt.ylabel(r'Depth Factor $r$', fontsize=14)
m_vals_upper = m_vals[m_vals > np.exp(d/2)]
plt.plot(m_vals_upper, np.log(m_vals_upper)/(d/2), 'y-', linewidth=3) # m**{1/r} = e**(d/2)
plt.tight_layout()
plt.savefig('heatmap_overthinking_theory.pdf', dpi=100, bbox_inches='tight')
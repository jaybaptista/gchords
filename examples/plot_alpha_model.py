# %%
import gchords
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patheffects as pe
# %%

mix_model_1 = gchords.GCSDornanMixture(scatter=0, alpha=0.2)
um = gchords.UniverseMachineMStarFit(scatter=0)

# Inset plot 

fig, ax = plt.subplots(dpi=400, figsize=(2,2))

masses = np.logspace(8, 12, 100)

zs = [0, 2, 5]
colors = ['k', 'r', 'k']
alphas = [0.2, 1, 0.2]
lws = [1, 3, 1]
for z in zs:
    smass = um.m_star(masses, z=z)
    ax.plot(np.log10(smass), np.log10(mix_model_1.mass(masses, z=z)), color=colors[zs.index(z)], alpha=alphas[zs.index(z)], linewidth=lws[zs.index(z)])


ax.text(8, 7.5, r'$z=0$', rotation=55, color='black', fontsize=12, weight='bold',
        path_effects=[pe.withStroke(linewidth=3, foreground='white')])
ax.text(10, 8.1, r'$z=2$', rotation=50, color='black', fontsize=12, weight='bold',
        path_effects=[pe.withStroke(linewidth=3, foreground='white')])
ax.text(6, 5.45, r'$z=5$', rotation=40, color='black', fontsize=12, weight='bold',
        path_effects=[pe.withStroke(linewidth=3, foreground='white')])
ax.set_xlabel(r'$\log_{10} (M_\star / M_\odot)$', fontsize=18)
ax.set_ylabel(r'$\log_{10} (M_\mathrm{gcs} / M_\odot)$', fontsize=18)

ax.set_xlim(3.5, 12)
ax.set_ylim(5, 9)

plt.show()
# %%
# Figure 1 — GC halo connection plot
fig, ax = plt.subplots(3, 1, dpi=400, figsize=(4,7), sharex=True, sharey=True)

alphas = [0, 0.5, 1]
zs = [0, 2, 5]
colors = ['xkcd:steel blue', 'xkcd:blurple', 'xkcd:coral']
linestyles = ['-', '--', ':']

for k in range(len(alphas)):
    alpha_val = alphas[k]
    mix_model = gchords.GCSDornanMixture(scatter=0, alpha=alpha_val, z_form_weight=True)

    for j in range(len(zs)):
        smass = um.m_star(masses, z=zs[j])
        ax[k].plot(np.log10(smass), np.log10(mix_model.mass(masses, z=zs[j])),
                   color=colors[j], linestyle=linestyles[j], alpha=1, linewidth=2)

    ax[k].set_xlabel(r'$\log_{10} (M_\star / M_\odot)$', fontsize=18)
    ax[k].set_ylabel(r'$\log_{10} (M_\mathrm{gcs} / M_\odot)$', fontsize=18)
    ax[k].set_xlim(3.5, 10)
    ax[k].text(0.1, 0.85, r'$\alpha={}$'.format(alpha_val), transform=ax[k].transAxes, fontsize=18, weight='bold',
            path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    ax[k].grid(alpha=0.2, linewidth=1)
ax[2].legend([r'$z=0$', r'$z=2$', r'$z=5$'], loc='lower right', fontsize=14)
fig.subplots_adjust(hspace=0.)
fig.savefig('plot_alpha_model.pdf', dpi=400, bbox_inches='tight')
plt.show()
# %%

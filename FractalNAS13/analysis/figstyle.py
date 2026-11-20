"""One palette, one set of rcParams, one pair of column widths, for every figure.

Keeping these here rather than in a figure script is what makes the four main figures and the
supplement's three read as one set, and it is the only place to change a colour or a type size.

The palette is categorical and was checked for the three common colour-vision deficiencies:
the two dataset colours differ in lightness as well as in hue, so a greyscale print of any
figure still separates CIFAR-10 from CIFAR-100.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import common

FIG = os.path.join(common.HERE, '..', 'figures')

C = {'c10': '#1f5fa8', 'c100': '#c9452c', 'good': '#1f8a70', 'alt': '#8a6bbf',
     'grey': '#6b6b6b', 'light': '#d8d8d4'}

plt.rcParams.update({
    'font.size': 7.2, 'axes.labelsize': 7.2, 'axes.titlesize': 7.6,
    'xtick.labelsize': 6.6, 'ytick.labelsize': 6.6, 'legend.fontsize': 6.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.2,
    'legend.frameon': False, 'figure.dpi': 160, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01, 'axes.grid': True, 'grid.linewidth': 0.4,
    'grid.color': '#e8e8e4', 'axes.axisbelow': True, 'pdf.fonttype': 42,
})

#: one- and two-column widths in inches, for IEEEtran's journal layout
W1, W2 = 3.4, 7.1


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    fig.savefig(os.path.join(FIG, name + '.png'))
    plt.close(fig)
    print('wrote figures/' + name + '.pdf')


def J(n):
    """One of the machine-readable result files every figure is drawn from."""
    return json.load(open(os.path.join(common.RES, n + '.json')))

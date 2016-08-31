#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import argparse
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


parser = argparse.ArgumentParser()
parser.add_argument('infiles', nargs='+')
parser.add_argument('--outfile')
parser.add_argument('--title')
parser.add_argument('--xlabel', default=r'$x$')
parser.add_argument('--ylabel', default=r'$y$')
parser.add_argument('--top', type=float)
parser.add_argument('--bottom', type=float)
parser.add_argument('--aspect', type=float)
parser.add_argument('--scale', default='linear')
args = parser.parse_args()

# data
alldata = []
mus = []
sigmas = []
for argsInfile in args.infiles:
    with open(argsInfile, 'r') as infile:
        filedata = []
        reader = csv.reader(infile)
        for row in reader:
            filedata.append(map(lambda s: int(s), row))
        alldata.append(filedata)
        mus.append(map(lambda arr: np.mean(arr), filedata))
        sigmas.append(map(lambda arr: np.std(arr), filedata))

# upper and limits
mu = reduce(lambda acc, x: acc + x, mus)
sigma = reduce(lambda acc, x: acc + x, sigmas)
ks = range(4,31) # XXX
if args.bottom:
    ymin = args.bottom
else:
    ymin = min(map(lambda x, y: x-y, mu, sigma))
if args.top:
    ymax = args.top
else:
    ymax = max(map(lambda x, y: x+y, mu, sigma))

# figure and axes
if args.aspect:
    w, h = plt.figaspect(args.aspect)
    fig = plt.figure(figsize=(w,h))
else:
    fig = plt.figure()
xlim = (min(ks), max(ks))
ylim = (ymin, ymax)
ax = fig.add_subplot(1, 1, 1, xlim=xlim, ylim=ylim)

def mixer1(a, b):
    alpha = 1.0-mix
    return a*alpha + b*(1.0-alpha)

mix = 0.25

# interpolate data
n_smooth = np.linspace(min(ks), max(ks), 1024)

# one
one_mu_1 = (interp1d(ks, mus[0], kind='slinear'))(n_smooth)
one_sigma_1 = (interp1d(ks, sigmas[0], kind='slinear'))(n_smooth)
one_sigma_2 = (interp1d(ks, sigmas[0], kind='cubic'))(n_smooth)
one_sigma_3 = map(mixer1, one_sigma_1, one_sigma_2)
one_top = map(lambda x, y: x+y, one_mu_1, one_sigma_3)
one_bot = map(lambda x, y: x-y, one_mu_1, one_sigma_3)

# two
two_mu_1 = (interp1d(ks, mus[1], kind='slinear'))(n_smooth)
two_sigma_1 = (interp1d(ks, sigmas[1], kind='slinear'))(n_smooth)
two_sigma_2 = (interp1d(ks, sigmas[1], kind='cubic'))(n_smooth)
two_sigma_3 = map(mixer1, two_sigma_1, two_sigma_2)
two_top = map(lambda x, y: x+y, two_mu_1, two_sigma_3)
two_bot = map(lambda x, y: x-y, two_mu_1, two_sigma_3)

# one
ax.fill_between(n_smooth, one_bot, one_top, facecolor="#7f7fff", alpha=0.75, zorder=2.0)
ax.plot(ks, mus[0], label=args.infiles[0], color="#0000ff", linewidth=3, zorder=2.5)

# two
ax.fill_between(n_smooth, two_bot, two_top, facecolor="#7f7f7f", alpha=0.75, zorder=1.0)
ax.plot(ks, mus[0], label=args.infiles[1], color="#16161d", linewidth=3, zorder=1.5)

# labeling and what-not
ax.set_ylabel(args.ylabel, fontdict={'size': 22})
ax.set_xlabel(args.xlabel, fontdict={'size': 22})
ax.grid(True)
ax.legend(loc='best')
if args.title:
    ax.set_title(args.title, fontdict={'size': 24})

# output
if args.outfile:
    plt.savefig(args.outfile)
else:
    plt.show()

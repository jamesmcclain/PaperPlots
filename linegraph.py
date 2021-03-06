#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2002-2016, James McClain
# All rights reserved.

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
parser.add_argument('--infiles', required=True, nargs='+')
parser.add_argument('--outfile')
parser.add_argument('--title')
parser.add_argument('--xs', type=str)
parser.add_argument('--xlabel', default=r'$x$')
parser.add_argument('--ylabel', default=r'$y$')
parser.add_argument('--most', type=float)
parser.add_argument('--least', type=float)
parser.add_argument('--aspect', type=float)
parser.add_argument('--sigma', nargs='+')
parser.add_argument('--mu', required=True, nargs='+')
parser.add_argument('--names', required=True, nargs='+')
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
            filedata.append(map(lambda s: float(s), row))
        alldata.append(filedata)
        mus.append(map(lambda arr: np.mean(arr), filedata))
        sigmas.append(map(lambda arr: np.std(arr), filedata))

# xs
if args.xs:
    bounds = map(lambda n: int(n), args.xs.split(':'))
    xs = range(bounds[0], bounds[1])
else:
    xs = range(0,len(mus[0]))

# upper and limits
mu = reduce(lambda acc, x: acc + x, mus)
sigma = reduce(lambda acc, x: acc + x, sigmas)
if args.least:
    ymin = args.least
elif args.sigma == None:
    ymin = min(mu)
else:
    ymin = min(map(lambda x, y: x-y, mu, sigma))

if args.most:
    ymax = args.most
elif args.sigma == None:
    ymax = max(mu)
else:
    ymax = max(map(lambda x, y: x+y, mu, sigma))

# figure and axes
if args.aspect:
    w, h = plt.figaspect(args.aspect)
    fig = plt.figure(figsize=(w,h))
else:
    fig = plt.figure()
xlim = (min(xs), max(xs))
ylim = (ymin, ymax)
ax = fig.add_subplot(1, 1, 1, xlim=xlim, ylim=ylim)

def mixer1(a, b):
    alpha = 1.0-mix
    return a*alpha + b*(1.0-alpha)

mix = 0.25

# interpolate data
n_smooth = np.linspace(min(xs), max(xs), 1024)

# draw
for i in range(0,len(args.mu)):
    mu_1 = (interp1d(xs, mus[i], kind='slinear'))(n_smooth)
    sigma_1 = (interp1d(xs, sigmas[i], kind='slinear'))(n_smooth)
    sigma_2 = (interp1d(xs, sigmas[i], kind='cubic'))(n_smooth)
    sigma_3 = map(mixer1, sigma_1, sigma_2)
    most = map(lambda x, y: x+y, mu_1, sigma_3)
    least = map(lambda x, y: x-y, mu_1, sigma_3)
    if ((type(args.sigma) == type([])) and (len(args.sigma) > i)):
        ax.fill_between(n_smooth, least, most, facecolor=args.sigma[i], alpha=0.75, zorder=i)
    ax.plot(xs, mus[i], label=args.names[i], color=args.mu[i], linewidth=3, zorder=(i + 0.5))

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

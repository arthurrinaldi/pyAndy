#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 18:05:47 2018

@author: user
"""

import pyandy.core.plotpage as pltpg
import pandas as pd
import numpy as np
import itertools

import pyandy.auxiliary.aux_sql_func as aql



df_0 = aql.read_sql('storage2', 'out_cal_1_linonly', 'analysis_time_series',
                    filt=[('sta_mod', ['stats_agora']),
                          ('pwrerg_cat', ['pwr']),
                          ('dow_type', ['SUN', 'WEEKDAY'])])



# %%
'''
# Basic example

Without additional parameters the PlotTiled class allows to efficiently
arrange subplots based on dimensions specified by the index plot arguments 
(ind_pltx and ind_plty). Here the emphasis lies on efficient visualization
of data along multiple dimensions.

In the example below hourly electricityy production data is averaged by
hour of the day and shown by season and day of the week type (Sunday/weekday).
The kind_dict parameter allows to specify a different plot type for a certain
series. Additional arguments to PlotTiled are passed on to all 

The caption in the lower left corner provides basic information on the 
underlying data. Most notably, it prints the n_min and n_max attributes of the 
PlotPageData instance, corresponding to the maximum and minimum number of data
points aggregated in the plot.
'''


from importlib import reload
import matplotlib.pyplot as plt

reload(pltpg)

do = pltpg.PlotPageData.from_df(df_0,
                                ind_pltx=['season'],
                                ind_plty=['dow_type'],
                                ind_axx=['hour'],
                                series=['fl'],
                                values=['value_posneg'],
                                data_scale={'dmnd': -1})

page_kws = dict(left=0.1, right=0.9, bottom=0., top=0.9,
                sharex=True, sharey=False)#, page_dim=(5,3), dpi=100)
label_kws = dict(label_format=' ', label_subset=[-1])
plot = pltpg.PlotTiled(do, kind_def='StackedArea', 
                       kind_dict={'dmnd': 'LinePlot'},
                       linewidth=2, marker='o',
                       **page_kws, **label_kws)


# %%


'''
# Colormaps

The colormap parameter can be a dictionary, a matplotlib colormap
plt.get_cmap('Set2') or a single color hex value.
'''

colormap = {'hard_coal': '#FF1F00',
            'lignite': '#59221B',
            'natural_gas': '#7F5200',
            'nuclear_fuel': '#E5853B',
            'wind_offshore': '#00027F',
            'wind_onshore': '#4C4FFF',
            'photovoltaics': '#4CEFFF',
            'hydro_total': '#D8C5EA',
            'pumped_hydro': '#D9D9D8',
            'bio_all': '#3B662E',
            'others': 'green',
            'dmnd': 'k'}

plot = pltpg.PlotTiled(do, kind_def='StackedArea', 
                       kind_dict={'dmnd': 'LinePlot'},
                       colormap=colormap,
                       linewidth=5, marker='o',
                       **page_kws, **label_kws)

# %

for ix, namex, iy, namey, plot, ax, kind in plot.get_plot_ax_list():
    
    ax.set_title('')











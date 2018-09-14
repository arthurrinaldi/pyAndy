# -*- coding: utf-8 -*-

from importlib import reload
import numpy as np
import pandas as pd

import grimsel_h.plotting.plotpage as pltpg

reload(pltpg)

n = 300

df = pd.DataFrame(np.random.randn(n, 1), columns=['value'])
df['bar'] = np.random.choice(['bar_a', 'bar_b'], n)
df['series'] = np.random.choice(['series_1', 'series_2', 'series_3'], n)
df['ind_axx'] = np.random.choice(['x_1', 'x_2', 'x_4', 'x_5'], n)
df['ind_pltx'] = np.random.choice(['pltx_1', 'pltx_2'], n)
df['ind_plty'] = np.random.choice(['plty_1', 'plty_2'], n)


do = pltpg.PlotPageData.from_df(df, ['ind_pltx'], ['ind_plty'], ['ind_axx'],
                           ['value'], ['series'], ind_axy=['bar'], aggfunc=sum)
do.data = do.data.fillna(1)

layout_kws = dict(bottom=0.1, top=0.9, left=0.1, right=0.99, page_dim=(7,4))
plots = pltpg.PlotTiled(do, kind_def='StackedGroupedBar',
                        barwidth=0.35, barspace=0.4,
                        xticklabel_rotation=0,
                        xlabel='x_label',
                        sharex=True, **layout_kws,
                        )

for indx, namex, indy, namey, plot, ax, kind in plots.get_plot_ax_list():

    title = namex[0] + namey[0]
    ax.set_title('%s, %s'%(namex[0], namey[0]))
    
    if indx == 0:
        ax.set_ylabel('y_label')
    else:
        ax.set_ylabel('')
        
    if indy == len(plots.list_ind_plty) - 1:
        ax.set_xlabel('x_label')
    else:
        ax.set_xlabel('')
    

# %%
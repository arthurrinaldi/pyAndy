# pyAndy
pyAndy enjoys arranging graphical elements in regular grids


TODO:
* wrap mticker locators
* generalize parameter dicts with custom parameters for each plot
* fix legends (including order of entries)
* wrap reindex/combine with series_order
* automate add_subplotletter
* check axes_rotation implementation
* implement shared axis labels, like so:

ax0 = plt_all_all.plotdict[(('yr2008',), ('bubbles',), 'BubblePlot')].ax
ax1 = plt_all_all.plotdict[(('yr2016',), ('bubbles',), 'BubblePlot')].ax

xlabel_pos = 0.5 * (ax0.transAxes.transform((1,0)) + ax1.transAxes.transform((0,0)))
xlabel_pos = ax0.transAxes.inverted().transform(xlabel_pos)
xlabel_pos[1] -= x_label_offset

ax0.set_xlabel('Charging hour')
ax0.xaxis.set_label_coords(*xlabel_pos)

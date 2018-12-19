# pyAndy
pyAndy enjoys arranging graphical elements in regular grids


TODO:
* wrap mticker locators
* generalize parameter dicts with custom parameters for each plot
* fix legends (including order of entries)
* wrap reindex/combine with series_order
* automate add_subplotletter
* check axes_rotation implementation
* support for hatches
* implement shared axis labels, like so:

```python
ax0 = plt.plotdict[...].ax
ax1 = plt.plotdict[...].ax

xlabel_pos = 0.5 * (ax0.transAxes.transform((1,0)) + ax1.transAxes.transform((0,0)))
xlabel_pos = ax0.transAxes.inverted().transform(xlabel_pos)
xlabel_pos[1] -= x_label_offset

ax0.set_xlabel('x_label')
ax0.xaxis.set_label_coords(*xlabel_pos)
```

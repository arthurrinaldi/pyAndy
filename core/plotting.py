'''
Basic plotting module adapted to the requirements of typical GRIMSEL
analyses.
Don't expect aesthetically pleasing results.
Largely a convenience wrapper for matplotlib and pandas plotting methods.

Inheritance scheme:

PlotsBase
    ├── StackedBase
    │    └── StackedGroupedBar
    ├── GroupedPlot <-- as provided by pandas
    ├── StackedPlot <-- as provided by pandas
    ├── AreaPlot <-- as provided by pandas
    ├── LineBase
    │    ├── LinePlot
    │    └── StepPlot
    ├── BoxPlot <-- as provided by seaborn
    └── WaterFallChart
'''

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.transforms import Affine2D
from matplotlib.collections import PathCollection

import seaborn as sns

from pyAndy.core.plotdata import PlotData

def new_update(self, props):
    '''
    This is a monkey patch for the update method of the matplotlib
    Artist class. We deactivate the error raised in case unknown parameters
    are provided. This allows us to construct
    plots with arbitrary input parameters, of which the meaningless ones get
    ignored.
    '''

    def _update_property(self, k, v):
        k = k.lower()
        if k in {'axes'}:
            return setattr(self, k, v)
        else:
            func = getattr(self, 'set_' + k, None)
            if not callable(func):
                pass
            else:
                # DELETED: raise AttributeError('Unknown property %s' % k)
                return func(v)

    store = self.eventson
    self.eventson = False
    try:
        ret = [_update_property(self, k, v)
               for k, v in props.items()]
    finally:
        self.eventson = store

    if len(ret):
        self.pchanged()
        self.stale = True
    return ret

import matplotlib
matplotlib.artist.Artist.update = new_update

class Meta(type):
    '''
    The Meta class is used to call methods after any child's __init__ is done.
    This is used to:
        - Manipulate axes properties after the creation of the plot.
        - Limit the instantiation to the definition of the plot attributes
            without drawing the plots themselves.
    '''

    def __call__(cls, *args, **kwargs):
        plt_inst = super().__call__(*args, **kwargs)
        # parent class method called after all childrens' __init__s

        if plt_inst.draw_now:
            plt_inst.draw_plot()
        return plt_inst

class PlotsBase():#metaclass=Meta):
    ''' Abstract class with basic plotting functionalities. '''

    def _update_serieskws(self, series_name=None):
        '''
        Updates the kwargs for the plot object.

        This is called for every series to adjust the properties.
        (Exception: PlotPandas)
        '''

        defaults = {'x_offset': 0,
                    'y_offset': False,
                    'xlabel': '',
                    'ylabel': '',
                    'on_values': False,
                    'bar': [1, 1],
                    'colormap': False,
                    'linestyle': '-',
                    'colorpos': 'green',
                    'colorneg': 'red',
                    'markersize': mpl.rcParams['lines.markersize'],
                    'opacitymap': 1,
                    'linewidth': None,
                    'width': False,
                    'bins': 10,
                    'ylim': False,
                    'xlim': False,
                    'edgecolor': None,
                    'label_format': False,#"{0:.1f}",
                    'label_threshold': 0,
                    'loc_labels': 0.5,
                    'label_angle': 0,
                    'label_subset': False,
                    'label_ha': 'left',
                    'bar_label_ypos': 0,
                    'barname_angle': 0,
                    'barname': None,
                    'groupname': None,
                    'xticklabels': False,
                    'title': '',
                    'marker': None,
                    'markerfacecolor': 'white',
                    'stacked': True,
                    'reset_xticklabels': False,
                    'offset_col': False,
                    'offs_slct': False,
                    'bubble_scale': 1,
                    'edgewidth': 0,
                    'barwidth': 0.9,
                    'barspace': False,
                    'xticklabel_rotation': 90,

                    # for BoxPlot
                    'show_outliers': False,
                    'draw_now': True,
                    'axes_rotation': False,
                    'gridpos': (0, 0),
                    'seriesplotkws': {},
                    'step': 'post',
                    }


        _kw_tot = dict()
        if not series_name and not self.plotkwargs:
            # initial parameter setting from non-dict-dict plotkwargs
            _kw_tot.update(self.kwargs)

        elif self.plotkwargs and not series_name:
            # initial parameter setting from dict_dict plotkwargs
            _kw_tot.update(self.plotkwargs[list(self.plotkwargs)[0]])
            _kw_tot.update(self.kwargs)
        elif self.plotkwargs and series_name:
            # update to specific series_name
            _kw_tot.update(self.plotkwargs[series_name])
            _kw_tot.update(self.kwargs)
        elif not self.plotkwargs and series_name:
            # do nothing
            pass

        if self.plotkwargs or not series_name:
            for key, val in defaults.items():
                setattr(self, key, val)
                if key in _kw_tot:
                    setattr(self, key, _kw_tot[key])
                    _kw_tot.pop(key)
                if key in self.kwargs:
                    self.kwargs.pop(key)



    def __init__(self, data, ax=None, plotkwargs=None, *args, **kwargs):
        '''
        Plotkwargs are
        '''

        if ax is None:
            _, self.ax = plt.subplots(1,1)
        else:
            self.ax = ax


        self.linedict = {}
        self.list_y_offset = False


        print(plotkwargs)
        if not plotkwargs:
            # plotkwargs not provided
            plotkwargs = {'__constant': kwargs.copy()}

        print(plotkwargs)

        self.plotkwargs = plotkwargs
        self.kwargs = kwargs
        self._update_serieskws()


        self.plotdata = PlotData(data, self.stacked, self.on_values)

        if not self.label_subset:
            self.label_subset = np.arange(len(self.plotdata.data))

        # initialize legend handles and labels for the plot
        self.pltlgd_handles = self.pltlgd_labels = []

        self.ibar = self.nbar = 1

        self.labels = {} # dict of all label objects

        self.generate_colors()
        self.generate_opacity()

        self.plotdata.init_xpos(barspace=self.barspace,
                                ibar=self.ibar,
                                nbar=self.nbar,
                                x_offset=self.x_offset)


    def gen_plot(self):
        '''
        Loops over plot series.
        '''

        iic, ic = list(enumerate(self.plotdata.c_list))[0]
        for iic, ic in enumerate(self.plotdata.c_list):

            if ic in self.plotkwargs:
                self._update_serieskws(ic)

            y = self.plotdata.get_series_y(ic)
            ic_color = self.plotdata.c_list_color[iic]

            self.linedict[ic] = self.gen_single_plot_series(ic, iic,
                                                            ic_color, y)

            self.adapt_plot_series(y)



    def _draw_data_labels(self, xpos=None):
        '''
        Adds labels to all plot series and selected x positions.

        The series name if included in the label in all cases.
        Label value formatting is set by the label_format attribute.
        The label_subset is a list of integer x-axis locations selecting
        the positions where a label is to be added.

        Args:
            xpos (iterable): x positions, set to default self.plotdata.xpos
                             if the parameter is not set; StackedGroupedBar
                             uses other values
        '''

        if xpos is None:
            xpos = self.plotdata.xpos

        for series, series_name in zip(self.plotdata.c_list,
                                       self.plotdata.c_list_names):

            y = self.plotdata.data[series].values

            labs = [self.label_format.format(iy)
                    if '{' in self.label_format.replace('{name}', '')
                    else self.label_format%iy
                    if abs(iy) > self.label_threshold else ''
                    for iy in y]

            if '{name}' in self.label_format:
                labs = [l.format(name=str(series_name)) if l != '' else '' for l in labs]

            offs = self.plotdata.data_offset[series].values

            for it in self.label_subset:

                label_pos = (xpos[it], (offs[it] + self.loc_labels * y[it]))

                lab = self.ax.annotate(labs[it], xy=label_pos, xytext=label_pos,
                                       xycoords='data',textcoords='data',
                                       horizontalalignment=self.label_ha,
                                       verticalalignment='center',
                                       rotation=self.label_angle)

                self.labels.update({(labs[it], it): lab})

    def draw_data_labels(self):
        '''
        Interface method calling _draw_data_labels.

        This method is replaced by StackedGroupedBar, which uses different
        xpos for the _draw_data_labels call.
        '''

        self._draw_data_labels()

    @property
    def gridpos(self):

        return self._gridpos

    @gridpos.setter
    def gridpos(self, value):
        '''
        We want to keep the gridpos from the first initialization as a
        private attribute, no matter what happens to the instance after that.
        '''

        if not hasattr(self, '_gridpos'):
            self._gridpos_0 = value

        self._gridpos = value


    def reset_gridpos(self):

        self._gridpos = self._gridpos_0


    def reset_legend_handles_labels(self):
        self.pltlgd_handles = []
        self.pltlgd_labels = []

    def draw_plot(self):

        self.reset_legend_handles_labels()
        self.gen_plot()

        if self.label_format:
            self.draw_data_labels()

        self.finalize_axis()


    def generate_colors(self):
        '''
        We require colormaps formatted as dictionaries. They are
        generated here, if not provided in the kwargs.
        '''
        if (type(self.colormap) == dict): # color dictionary
            self.colors = self.colormap
        elif (type(self.colormap) == str): #defined constant color
            self.colors = {self.plotdata.c_list_color[i]: self.colormap
                           for i in range(len(self.plotdata.c_list_color))}
        else: #built-in colors
            if self.colormap:
                cmap = self.colormap
            else:
                cmap = plt.get_cmap('Set2')
            self.colors = {series:
                           cmap(iseries/float(len(set(self.plotdata.c_list_color))))
                           for iseries, series
                           in enumerate(set(self.plotdata.c_list_color))}

    def generate_opacity(self):
        ''' Same as colors, but for opacity. '''
        if type(self.opacitymap)==dict:
            self.opacity = self.opacitymap;
        else:
            self.opacity = {cc: self.opacitymap for cc in self.plotdata.c_list_color}


    def get_legend_handles_labels(self):
        '''
        Get legend handles and labels for the whole axes.

        If these are created manually (such as in StackedArea), the method is
        overwritten in the respective child class.
        '''

        return self.ax.get_legend_handles_labels()


    def _reset_xticklabels(self):

        self.ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(len(self.plotdata.data))))
        self.ax.xaxis.set_major_formatter(mticker.FixedFormatter(self.plotdata._data.index.tolist()))
        plt.setp(self.ax.get_xticklabels(), rotation=self.xticklabel_rotation)


    def finalize_axis(self):

        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)

        if self.axes_rotation:
            self.rotate_axes()

        if self.reset_xticklabels:

            self._reset_xticklabels()

        if not (type(self.ylim) == bool and self.ylim == False):
            if type(self.ylim) in [list, tuple]:
                self.ylim = {bt: self.ylim[nbt]
                             for nbt, bt in enumerate(['bottom', 'top'])}
            self.ax.set_ylim(**self.ylim)

        if not (type(self.xlim) == bool and self.xlim == False):
            if type(self.xlim) in [list, tuple]:
                self.xlim = {bt: self.xlim[nbt]
                             for nbt, bt in enumerate(['left', 'right'])}
            self.ax.set_xlim(**self.xlim)

    def adapt_plot_series(self, y):
        '''
        Implemented by some children. Or not.
        '''

    def rotate_axes(self):

        ylim_0 = self.ax.get_ylim()
        xlim_0 = self.ax.get_xlim()

        lim_exp = [max(ylim_0 + xlim_0)] * 2
        lim_exp[0] *= -1

        self.ax.set_xlim(lim_exp)
        self.ax.set_ylim(lim_exp)

        r = Affine2D().rotate_deg(self.axes_rotation)
        r.set_matrix(np.matmul(r.get_matrix(), np.diag([1,-1,1])))

        for x in self.ax.images + self.ax.lines + self.ax.collections:
            trans = x.get_transform()
            x.set_transform(r+trans)
            if isinstance(x, PathCollection):
                transoff = x.get_offset_transform()
                x._transOffset = r + transoff

        self.ax.set_xlim(ylim_0)
        self.ax.set_ylim(xlim_0)

    def gatherAttrs(self):
        attrs = '\n'
        for key in self.__dict__:
            attrs += '\t%s=%s\n' % (key, self.__dict__[key])
        return attrs

    def __str__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.gatherAttrs( ))

    def reset_offset(self, data):
        ''' Implemented in StackedBase. '''


    def add_plot_legend(self, from_ax=False, handles=None, labels=None,
                        keep_n=None, string_replace_dict=None,
                        translate_dict=None, **legend_kwargs):
        '''
        Adds a legend to the this plot instance's axes.

        get_legend_handles_labels obtains the legend item handles and labels
        from the appropriate method. In the default case this is the
        PlotsBase method. In case of some children classes it is overwritten
        (e.g. StackedArea).

        Args:
            keep_n (int): index of series name to be kept as a legend label
            string_replace_dict (dict): dictionary for multiple String.replace
                                        operations on labels
            translate_dict (dict): dictionary with (final) labels as keys
        '''


        if not handles and not labels:
            if from_ax:
                hdls_lbls = self.ax.get_legend_handles_labels()
            else:
                hdls_lbls = self.get_legend_handles_labels()
        else:
            hdls_lbls = (handles, labels)

        hdls_lbls = (hdls_lbls[0], list(map(str, hdls_lbls[1])))

        print('legend_kwargs in add_plot_legend', legend_kwargs, 'keep_n', keep_n)

        if isinstance(keep_n, int):
            hdls_lbls = list(zip(*[(hh, ll[keep_n])
                                   for hh, ll in zip(*hdls_lbls)]))

        hdls, lbls = hdls_lbls

        if string_replace_dict:
            mrep = lambda s, d: (s if len(d) is 0
                                else mrep(s.replace(*d.popitem()), d))

            print('string_replace_dict: ', string_replace_dict)
            lbls = [mrep(str(ll), string_replace_dict.copy())
                    for ll in lbls]

        if translate_dict:
            lbls = [translate_dict[ll] if ll in translate_dict else ll
                    for ll in lbls]


        self.ax.legend(handles=hdls, labels=lbls, **legend_kwargs)


class BubblePlot(PlotsBase):
    '''
    Bubble plots require input DataFrame of shape:

                       bubble_size
                      data1   data2   ...
    index_x, index_y  -----   -----   ...
    '''

    def __init__(self, *args, **kwargs):

        kwargs.update({'on_values': True})

        super(BubblePlot, self).__init__(*args, **kwargs)



    def gen_plot(self):
#
#        xpos_dict = {vv: ii for ii, vv
#                     in enumerate(self.plotdata.data.index.get_level_values(0)
#                                                 .unique().tolist())}


#        fig, self.ax = plt.subplots(1,1)


        for iic, ic in enumerate(self.plotdata.c_list):
            data_slct = self.plotdata.data[ic]

            data_slct = data_slct.reset_index()

            s = [iis * self.bubble_scale for iis in data_slct[ic].tolist()]
            x = data_slct[self.plotdata.data.index.names[0]].tolist()
            y = data_slct[self.plotdata.data.index.names[1]].tolist()


            color = self.colors[self.plotdata.c_list_color[iic]]
            opacity = self.opacity[self.plotdata.c_list_color[iic]]


#            self.gen_single_plot_series(x, y, s, alpha, )

            self.ax.scatter(x=x, y=y, s=s, alpha=opacity, label=ic,
                            color=color);

#        self.reset_xticklabels = True


#        self.xtickvals = self.data.index.get_level_values(0).unique()
#        self.ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(len(self.xtickvals))))
#        self.ax.xaxis.set_major_formatter(mticker.FixedFormatter(self.xtickvals))



    def finalize_axis(self):
        ''' Things are different for bubbleplots.'''

        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)

        self.xtickvals = self.plotdata.data.index.get_level_values(0).unique()

        if self.reset_xticklabels and not self.barname:

            self.ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(len(self.xtickvals))))
            self.ax.xaxis.set_major_formatter(mticker.FixedFormatter(self.xtickvals))

class PlotPandas(PlotsBase):
    '''
    Makes use of the pandas plot method.
    This overwrites the gen_plot as we don't need to loop over series.
    '''

    def __init__(self, data, pd_method, *args, **kwargs):

        self.on_values=True

        super(PlotPandas, self).__init__(data, *args, **kwargs)

        if 'bar' in pd_method:
            # bar plots on integer indices, not on values; need to set
            # on_values to False, otherwise the labels are not aligned
            self.on_values = False
            self.plotdata.on_values = False
            self.plotdata.init_xpos()

        self.pd_method = pd_method


    def get_color_list(self):
        '''
        Pandas expects a list of colors corresponding to the data series.
        '''

        color_list = [self.colors[c] for c in self.plotdata.c_list_color]
        if len(color_list) <= 1:
            color_list = color_list[0]

        return color_list

    def gen_plot(self):

        _pd_method = self.pd_method.split('.')

        obj = self.plotdata.data
        for imeth in _pd_method:
            obj = getattr(obj, imeth)

        _plotfunc = obj


        kwargs = dict(ax=self.ax, marker=self.marker,
                      stacked=self.stacked, lw=self.linewidth,
                      legend=False, alpha=self.opacitymap,
                      edgecolor=self.edgecolor,
                      markersize=self.markersize,
                      color=self.get_color_list(), width=self.barwidth)

        # for hist
        if self.pd_method == 'plot.hist':
            kwargs.pop('width')

            kwargs.update(dict(rwidth=self.barwidth,
                               bins=self.bins))

        # all pandas plotting method
        _plotfunc(**kwargs)

class StackedBase(PlotsBase):
    '''
    Handles everything that's relevant for stacked plots. Serves as abstract
    class for
        - stacked area and
        - stacked grouped bar plots.

    APPEARS TO BE OBSOLETE SINCE SWITCH TO PlotData
    '''
    def __init__(self, *args, **kwargs):


        super(StackedBase, self).__init__(*args, **kwargs)

class StackedArea(StackedBase):
    ''' Stacked area plot. '''
    def __init__(self, *args, **kwargs):

        super(StackedArea, self).__init__(*args, **kwargs)

    def gen_single_plot_series(self, ic, ic_name, ic_color, y):
        ''' Use stacked fill_between to add areas to stacked area plot. '''

        offs = self.plotdata.data_offset[ic].values

        if self.step == 'post':
            dx = (self.plotdata.xpos[2] - self.plotdata.xpos[1])
            xpos = [pos - 0.5 * dx for pos in self.plotdata.xpos] + [self.plotdata.xpos[-1] + 0.5 * dx] #copy
            y = np.append(y, y[-1])
            offs = np.append(offs, offs[-1])
        else:
            xpos = self.plotdata.xpos


        if self.edgewidth == 0 or (not self.edgewidth):
            edge_opacity = 0
            edgecolor = None
            plot_line = [None]
        else:
            edge_opacity = self.opacity[ic_color]
            edgecolor = self.edgecolor

            plot_line = self.ax.step(xpos, offs + y,
                                    marker=None,
                                    linewidth=self.edgewidth,
                                    alpha=edge_opacity,
                                    color=edgecolor,
                                    where='post',)

        plot_area = self.ax.fill_between(xpos, offs, offs + y,
                             color=self.colors[ic_color],
                             alpha=self.opacity[ic_color],
                             linewidth=0,
                             label=ic, step=self.step,
                             )

        self.pltlgd_handles.append(mpatches.Patch(color=self.colors[ic_color],
                                                  alpha=self.opacity[ic_color]))
        self.pltlgd_labels.append(ic)

        return (plot_line[0], plot_area)

    def get_legend_handles_labels(self):
        '''
        Overwrites the parent method. Return manual legend items instead of
        from self.ax.
        '''

        return (self.pltlgd_handles, self.pltlgd_labels)


class StackedGroupedBar(StackedBase):
    '''
    Grouped bars are assembled through multiple calls to StackedGroupedBar
    using the keyword argument bar=[ibar, nbars]
    '''
    def __init__(self, *args, **kwargs):

        self.on_values = True
        self.stacked = True

        super(StackedGroupedBar, self).__init__(*args, **kwargs)

        self.reset_xticklabels = True

        # bars correspond to ind_axy
        self.list_bars = (self.plotdata.data.index
                              .get_level_values(-1).unique().tolist())
        self.nbar = len(self.list_bars)

        # bars require widths
        if not self.barwidth:
            self.barwidth = 1 / (self.nbar + 1)

        if not self.barspace:
            self.barspace = 1. / (self.nbar * 1.1 )



    def finalize_axis(self):
        super().finalize_axis()
#        self.gen_barname()

    def gen_barname(self):
        ''' Add a barname to the bar group, if provided in the kwargs. '''
        if self.barname:
            for ipos in self.xpos:
                self.ax.annotate(self.barname, xy=(0,0),
                                 xytext=(ipos, self.bar_label_ypos),
                                 xycoords='data',textcoords='data',
                                 horizontalalignment='center',
                                 verticalalignment='top',
                                 rotation=self.barname_angle, **self.kwargs)
            self.ax.set_xticks([])

    def gen_plot(self):
        '''
        Loops over plot series.
        '''

        self.reset_legend_handles_labels()

        self.xpos_minor = [] # save for xticks later

        for ibar, bar_name in enumerate(self.list_bars):


            self.ibar = ibar + 1 # bars are 1-indexed
            self.bar_name = bar_name

            data_bar = self.plotdata.data.xs(self.bar_name, axis=0, level=-1)
            offs_slct = self.plotdata.data_offset.xs(self.bar_name, axis=0,
                                                     level=-1)

            # update xpos due to bar selection
            self.plotdata.init_xpos(data=data_bar, barspace=self.barspace,
                                    ibar=self.ibar, nbar=self.nbar,
                                    x_offset=self.x_offset, on_values=False)

            self.xpos_minor += list(self.plotdata.xpos)

            print('xpos', self.plotdata.xpos)

            for iic, ic in enumerate(self.plotdata.c_list):

                y = data_bar[ic].values
                offs = offs_slct[ic].values
                ic_color = self.plotdata.c_list_color[iic]

                args = ic, iic, ic_color, y, offs
                self.linedict.update({ic: self.gen_single_plot_series(*args)})

#            self.adapt_plot_series(y)

    def draw_data_labels(self):
        '''
        Interface method calling _draw_data_labels. Overwrites the parent method.
        '''

        self._draw_data_labels(xpos=sorted(self.xpos_minor))




    def gen_single_plot_series(self, ic, ic_name, ic_color, y, offs):
        ''' Generate simple bar plot with offsets. The grouping is handled
            through the xpos.'''

        plot = self.ax.bar(self.plotdata.xpos, y,
                            edgecolor=self.edgecolor,
                            color=self.colors[ic_color],
                            alpha=self.opacity[ic_color],
                            bottom=offs,
                            align='center',
                            linewidth=self.linewidth,
                            width=self.barwidth,
                            label=ic)

        return plot[0]

    def _reset_xticklabels(self):
        '''
        For StackedGroupedBar plots the xticklabels don't include the bar names
        '''

        self.ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(len(self.plotdata.data))))
        self.ax.xaxis.set_major_formatter(mticker.FixedFormatter(self.plotdata._data.xs(self.plotdata._data.index.get_level_values(-1)[0], level=-1).index))
        plt.setp(self.ax.get_xticklabels(), rotation=self.xticklabel_rotation)

        self.ax.xaxis.set_minor_locator(mticker.FixedLocator(self.xpos_minor))
        self.ax.xaxis.set_minor_formatter(mticker.FixedFormatter(np.repeat(self.list_bars, len(self.plotdata.xpos))))
        plt.setp(self.ax.get_xminorticklabels(), rotation=self.xticklabel_rotation)

        self.ax.tick_params(axis='x', direction='in', which='minor', pad=-15)

        for tick in self.ax.xaxis.get_minor_ticks():
            tick.label1.set_verticalalignment('bottom')

# %


class LineBase(PlotsBase):
    ''' All we need for the subsequent definition of line-based plotting
        classes. '''
    def __init__(self, *args, **kwargs):

        kwargs['stacked'] = False

        PlotsBase.__init__(self, *args, **kwargs)
        self.loc_labels = self.loc_labels if self.stacked else 1

    def gen_single_plot_series(self, ic, y):
        '''Plot single series.'''

class LinePlot(LineBase):
    ''' Line plots. '''

    def gen_single_plot_series(self, ic, ic_name, ic_color, y):

        markerfacecolor = self.colors[ic_color] if not self.markerfacecolor else self.markerfacecolor

        plot = self.ax.plot(self.plotdata.xpos, y, color=self.colors[ic_color],
                     alpha=self.opacity[ic_color],
                     linewidth=self.linewidth,
                     marker=self.marker, label=ic,
                     markersize=self.markersize,
                     markerfacecolor=markerfacecolor,
                     linestyle=self.linestyle,
                     **self.kwargs)

        return plot[0]

class StepPlot(LineBase):
    ''' Step plots. '''
    def gen_single_plot_series(self, ic, ic_name, ic_color, y):
        plot = self.ax.step(self.plotdata.xpos, y, label=ic,
                            marker=self.marker,
                            linewidth=self.linewidth,
                            linestyle=self.linestyle,
                            markersize=self.markersize,
                            markerfacecolor=self.markerfacecolor,
                            alpha=self.opacity[ic_color],
                            color=self.colors[ic_color],
                            where='mid', **self.kwargs)

        return plot[0]

class WaterfallChart(PlotsBase):
    def __init__(self, *args, **kwargs):

        PlotsBase.__init__(self, *args, **kwargs)


    def gen_plot(self):

        if len(self.data.columns) > 1:
            df_wf = pd.DataFrame(self.data.sum(axis=1))
        else:
            df_wf = self.data.copy()

        df_wf = df_wf.iloc[:, 0]
        df_wf = df_wf[df_wf != 0].sort_values(ascending=False)

        self.offs_slct = df_wf.shift(1).fillna(0).cumsum().tolist()
        self.generate_xpos(df_wf)

        y = np.array(df_wf)
        self.gen_single_plot_series('', '', '', y)

        list_xticks = df_wf.index.get_values().tolist()
        self.ax.set_xticklabels([''] + list_xticks)

        self.ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(len(df_wf))))
        self.ax.xaxis.set_major_formatter(mticker.FixedFormatter(df_wf.index))


    def gen_single_plot_series(self, ic, ic_name, ic_color, y):
        ''' Generate simple bar plot with offsets. The grouping is handled
            through the xpos.'''

        plot = self.ax.bar(self.xpos, y,
                                edgecolor=self.edgecolor,
#                                color=self.colors[ic_color],
                                alpha=list(self.opacity.values())[0],
                                bottom=self.offs_slct,
                                align='center',
                                linewidth=self.linewidth,
                                width=self.barwidth,
                                label=''
                                )

        for ibar, bar in enumerate(self.plot):
            if y[ibar] > 0:
                bar.set_color(self.colorpos)
            else:
                bar.set_color(self.colorneg)

        return plot[0]

class BoxPlot(PlotsBase):

    def __init__(self, data, x, *args, **kwargs):
        '''
        Args:
            x:
        '''


        super(BoxPlot, self).__init__(data=data, *args, **kwargs)

        # stacking columns
        self.data = (self.plotdata.data.rename_axis('hue', axis=1)
                              .stack()
                              .rename('__value')
                              .reset_index())

        # sns.boxplot is picky with the hue columns
        make_str = lambda x: '({})'.format(', '.join([str(c) for c in x]
                                           if type(x) in (list, tuple)
                                           else [x]))
        self.data['hue'] = self.data['hue'].apply(make_str)

        self.box_plot_x = x
        self.box_plot_y = '__value'


    def gen_plot(self):
        '''
        Generate seaborn boxplot.

        This currently probably only works for single index x-axes.
        '''

        self.plot = sns.boxplot(x=self.box_plot_x, y=self.box_plot_y,
                                data=self.data, hue='hue',
                                palette='Set3', ax=self.ax,
                                showfliers=self.show_outliers)

        self.reset_xticklabels = True



def add_subplotletter(ax, n, ha='left', va='top', loc=(0, 1), offs=(5, -5), fs=None):
    list_abc = ['(%s)'%ii for ii in list('abcdefghijklmnopqrstuvwxyz')]

    if not fs:
        fs = ax.title.get_fontsize()

    t = ax.annotate(list_abc[n],
                    xy=loc, xycoords=ax.transAxes,
                    xytext=offs, textcoords='offset points',
                    fontsize=fs, va=va, ha=ha)
    t.set_bbox(dict(facecolor='white', linewidth=0, alpha=0.5,
                    boxstyle='square,pad=0'))
    t.set_zorder(1000)


# %%

#self.plotdata._data.index %%
if __name__ == '__main__':


#    data = do.data#.loc[('pwr')]
#
#    x = np.arange(2010, 2011, 0.1) + 100
#
#    data = pd.DataFrame(dict(year=x,
#                      sin=np.sin(x),
#                      const=np.ones(np.size(x)),
#                      const_neg=-np.ones(np.size(x)),
#                      tan=np.maximum(-2, np.minimum(2, np.tan(x)))),
#                        ).set_index('year')
#
#    data = pd.concat([data.assign(index='AAAAAAAAAAAAA', sin=data.sin * 0.2), data.assign(index='BBBBBBB', sin=data.sin*2), data.assign(index='C')])
#    data = data.set_index('index', append=-2)

    data = do.data.loc['DE0']

    self = StackedGroupedBar(data, pd_method='plot.area', stacked=True, opacitymap=0.9,
                      drawnow=True,

                      label_format='', on_values=True,
#                      label_subset=[-1,],
                      reset_xticklabels=False, label_ha='center',
                      xticklabel_rotation=90, linewidth=0, show_outliers=False,
                      edgecolor='k', edgewidth=5, bubble_scale=0.001,
                      loc_labels=0.5, axes_rotation=0,
                      barwidth=0.35, barspace=0.4)

    self.draw_plot()
    self.label_format = '\n%.4f'

#    self.add_plot_legend(ncol=2, loc=2,
#                         string_replace_dict={'value': '', ')': '', '(': ''},
#                         translate_dict={'wind_total': 'Total wind power'})
#






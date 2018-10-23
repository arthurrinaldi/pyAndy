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
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.transforms import Affine2D
from matplotlib.collections import PathCollection

import seaborn as sns

def new_update(self, props):
    '''
    This is a monkey patch for the update method of the matplotlib
    Artist class. We deactivate the error raised in case unknown parameters
    are provided. This allows us to construct
    plots with arbitrary input parameters, of which the meaningless ones get
    ignored.
    Note that this might be a terribly bad idea.
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

class PlotsBase(metaclass=Meta):
    ''' Abstract class with basic plotting functionalities. '''
    def __init__(self, data, ax=None, *args, **kwargs):

#        if not stacked:
#            diag_data = pd.Series(data.ix[:, 0])
#            data_2d = float('nan') * np.ones([len(diag_data), len(diag_data)])
#            np.fill_diagonal(data_2d, diag_data)
#
#            self.data = pd.DataFrame(data_2d,
#                                     index=data.index,
#                                     columns=data.index)
#        else:
        self.data = data

        if ax is None:

            _, self.ax = plt.subplots(1,1)
        else:
            self.ax = ax

        self.generate_c_list()

        self.linedict = {}
        self.list_y_offset = False

        defaults = {'x_offset': 0,
                    'y_offset': False,
                    'xlabel': '',
                    'ylabel': '',
                    'on_values': False,
                    'bar': [1, 1],
                    'colormap': False,
                    'colorpos': 'green',
                    'colorneg': 'red',
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
                    'label_ha': 'center',
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
                    'barwidth': False,
                    'barspace': False,
                    'markersize': None,
                    'xticklabel_rotation': 90,
                    # for BoxPlot
                    'show_outliers': False,
                    'draw_now': True,
                    'axes_rotation': False,
                    'gridpos': (0, 0),
                    'seriesplotkws': {},
                    }
        for key, val in defaults.items():
            setattr(self, key, val)
            if key in kwargs.keys():
                setattr(self, key, kwargs[key])
                kwargs.pop(key)

        if not self.label_subset:
            self.label_subset = np.arange(len(self.data))

        # what's left of kwargs is passed to the plotting functions
        self.kwargs = kwargs

        # initialize legend handles and labels for the plot
        self.reset_legend_handles_labels()

        self.ibar = self.nbar = 1

        self.labels = {} # dict of all label objects

        self.generate_xpos(self.data)
        self.generate_colors()
        self.generate_opacity()
#        self.generate_fonts()
#        self.finalize_axis()



    def set_gridpos(self, value):
        '''
        We want to keep the gridpos from the first initialization as a
        private attribute, no matter what happens to the instance after that.
        '''

        if '_gridpos' not in self.__dict__.keys():
            self._gridpos_0 = value

        self._gridpos = value

    def get_gridpos(self):

        return self._gridpos

    gridpos = property(get_gridpos, set_gridpos)

    def reset_gridpos(self):

        self._gridpos = self._gridpos_0


    def reset_legend_handles_labels(self):
        self.pltlgd_handles = []
        self.pltlgd_labels = []

    def draw_plot(self):

        self.gen_plot()
        self.finalize_axis()

    def generate_xpos(self, df):
        ''' List-like object for x-axis positions, either directly from data
            or generic. '''
        if self.on_values:
            self.xpos = df.index.get_values().tolist()
        else:
            barspace = 1 if not self.barspace else self.barspace

            self.xpos = (np.arange(len(df))
                         + self.x_offset
                         + (self.ibar - 0.5 * (self.nbar - 1) - 1)
                         * barspace)

    def get_series_y(self, ic):
        return np.array([iy for iy in self.data[ic].get_values()])

    def generate_colors(self):
        '''
        We require colormaps formatted as dictionaries. They are
        generated here, if not provided in the kwargs.
        '''
        if (type(self.colormap) == dict): # color dictionary
            self.colors = self.colormap
        elif (type(self.colormap) == str): #defined constant color
            self.colors = {self.c_list_color[i]: self.colormap
                           for i in range(len(self.c_list_color))}
        else: #built-in colors
            if self.colormap:
                cmap = self.colormap
            else:
                cmap = plt.get_cmap('Set2')
            self.colors = {series:
                           cmap(iseries/float(len(set(self.c_list_color))))
                           for iseries, series
                           in enumerate(set(self.c_list_color))}

    def generate_opacity(self):
        ''' Same as colors, but for opacity. '''
        if type(self.opacitymap)==dict:
            self.opacity = self.opacitymap;
        else:
            self.opacity = {cc: self.opacitymap for cc in self.c_list_color}

    def generate_c_list(self):
        ''' Data series are expected to be organized in columns. c_list is
            the list of columns.'''
        self.c_list = [c for c in self.data.columns]

        # c_list_names are used for indexing (colors etc). In case of
        # multiindex columns only the last element is selected.
        self.c_list_names = self.c_list.copy()
        if type(self.c_list[0]) in [list, tuple]:
            self.c_list_color = [cc[-1] for cc in self.c_list]

            # get relevant dimensions
            dims = []
            for idim in range(len(self.c_list[0])):
                if len(set([c[idim] for c in self.c_list])) > 1:
                    dims.append(idim)
            self.c_list_names = [list(c) for c in
                                 (np.array(self.c_list).T[dims].T)]
        else:
            self.c_list_color = self.c_list
            self.c_list_names = self.c_list

    def draw_data_labels(self, series_name, y, offs=False):
        """
        Add flexibly positionable data labels for all data points
        or a selection thereof
        """

        if type(offs) == bool and offs == False:
            offs = np.zeros(len(y))
        labs = [self.label_format.format(iy)
                if abs(iy) > self.label_threshold else '' for iy in y]

        labs = [str(series_name) + l if l != '' else '' for l in labs]

        for it in self.label_subset:
#            lab = self.ax.text(self.xpos[it],
#                         (offs[it] + self.loc_labels * y[it]),
#                         labs[it], ha=self.label_ha, va='center',
#                         rotation=self.label_angle)


            label_pos = (self.xpos[it], (offs[it] + self.loc_labels * y[it]))
            lab = self.ax.annotate(labs[it], xy=(0,0), xytext=label_pos,
                                   xycoords='data',textcoords='data',
                                   horizontalalignment=self.label_ha,
                                   verticalalignment='center',
                                   rotation=self.label_angle)


            self.labels.update({(labs[it], it): lab})

    def get_legend_handles_labels(self, keep_n=None):
        '''
        Get legend handles and labels for the whole axes.

        If these are created manually (such as in StackedArea), the method is
        overwritten in the respective child class.
        '''

        hdls_lbls = self.ax.get_legend_handles_labels()
        if isinstance(keep_n, int) or hasattr(keep_n, 'len'):
            hdls_lbls = list(zip(*[(hh, ll.split(', ')[keep_n])
                                   for hh, ll in zip(*hdls_lbls)]))

        return hdls_lbls


    def append_plot_legend_handles_labels(self):

        self.pltlgd_handles, self.pltlgd_labels = \
                                        self.get_legend_handles_labels()

    def _reset_xticklabels(self):

        self.ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(len(self.data))))
        self.ax.xaxis.set_major_formatter(mticker.FixedFormatter(self.data.index.tolist()))
        plt.setp(self.ax.get_xticklabels(), rotation=self.xticklabel_rotation)


    def finalize_axis(self):

        self.append_plot_legend_handles_labels()

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
                x._transOffset = r+transoff

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

    def gen_plot(self):
        '''
        Loops over plot series.
        '''
        self.reset_offset(self.data)
        self.reset_legend_handles_labels()

        for iic, ic in enumerate(self.c_list):
            y = self.get_series_y(ic)
            ic_name = self.c_list_names[iic]
            ic_color = self.c_list_color[iic]

            self.linedict.update({ic: self.gen_single_plot_series(ic, iic, ic_color, y)})

            if self.label_format:
                self.draw_data_labels(series_name=ic_name, y=y,
                                      offs=self.offs_slct)

            self.adapt_plot_series(y)

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

        xpos_dict = {vv: ii for ii, vv
                     in enumerate(self.data.index.get_level_values(0)
                                                 .unique().tolist())}


#        fig, self.ax = plt.subplots(1,1)


        for iic, ic in enumerate(self.c_list):
            data_slct = self.data[ic]

            data_slct = data_slct.reset_index()

            s = [iis * self.bubble_scale for iis in data_slct[ic].tolist()]
            x = data_slct[self.data.index.names[0]].tolist()
            y = data_slct[self.data.index.names[1]].tolist()


            color = self.colors[self.c_list_color[iic]]
            opacity = self.opacity[self.c_list_color[iic]]


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
#
#        handles, labels = self.ax.get_legend_handles_labels()
#        self.pltlgd_handles += handles
#        self.pltlgd_labels += labels
#
        self.xtickvals = self.data.index.get_level_values(0).unique()

        if self.reset_xticklabels and not self.barname:

            self.ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(len(self.xtickvals))))
            self.ax.xaxis.set_major_formatter(mticker.FixedFormatter(self.xtickvals))

#class BubblePlot(Plot3DBase):
#
#
#
#    self.ax.scatter(x=x, y=y, s=s, alpha=opacity, label=ic,
#                color=color);



class PlotPandas(PlotsBase):
    '''
    Makes use of the pandas plot method.
    This overwrites the gen_plot as we don't need to loop over series.
    '''
    def __init__(self, data, ax, pd_method, *args, **kwargs):

        self.on_values=True

        super(PlotPandas, self).__init__(data, ax, *args, **kwargs)
        self.pd_method = pd_method


    def gen_plot(self):

        _pd_method = self.pd_method.split('.')

        obj = self.data
        for imeth in _pd_method:
            obj = getattr(obj, imeth)

        print(self.data)

        _plotfunc = obj

        color_list = list(map(self.colors.get,
                              [c[-1] for c in self.data.columns]))
        if self.data.columns.size <= 1:
            color_list = color_list[0]

        kwargs = dict(ax=self.ax, marker=self.marker,
                      stacked=self.stacked, lw=self.linewidth,
                      legend=False, alpha=self.opacitymap,
                      edgecolor=self.edgecolor,
                      markerfacecolor=self.markerfacecolor,
                      markersize=self.markersize,
                      color=color_list, width=self.barwidth)

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
    '''
    def __init__(self, *args, **kwargs):


        super(StackedBase, self).__init__(*args, **kwargs)


        self.reset_offset(data=self.data)
        self.generate_c_list() # update in case the offset column got removed

    def get_list_y_offset(self):
        '''
        Save list_y_offset in attribute if it doesn't exist yet and delete
        the corresponding data column.
        '''

        if not type(self.list_y_offset) is bool:
            return self.list_y_offset
        else:
            list_y_offset = self.data[self.y_offset].copy().values
            self.data = self.data.drop(self.y_offset, axis=1)

            return list_y_offset


    def get_data_offset(self, data):
        '''
        If the parameter offset_col is not False, copy the corresponding
        data columns to the parameter data_offset.
        '''

        print(self.offset_col)
        print(data.columns)

#        if self.offset_col:
#            pass
#            self.data_offset = self.data[[c for c in data.columns
#                                          if self.offset_col
#                                          in c][0]].astype(float)
#
#            self.data = self.data[[c for c in data.columns
#                                   if not self.offset_col in c]]
#        else:

        self.data_offset = self.get_zero_offset(data)


    def get_zero_offset(self, data):

        return np.zeros(len(data))

    def init_offset(self, data):

        if self.offset_col:
            self.offs_pos = self.data_offset.copy()
        else:
            self.offs_pos = self.get_zero_offset(data)
        self.offs_neg = self.offs_pos.copy()

    def reset_offset(self, data):
        '''
        Input parameters:
        data -- DataFrame; will be self.data, or a subset thereof
        '''

        self.init_offset(data)

        if self.y_offset:

            self.list_y_offset = self.get_list_y_offset(data)

            self.offs_pos = self.list_y_offset
            self.offs_neg = self.list_y_offset

    def adapt_plot_series(self, y):
        ''' Update the offset after each data series. '''

        print('y in adapt_plot_series: ', y)
        self.offs_pos, self.offs_neg = self.set_offset(y, self.offs_pos,
                                                       self.offs_neg)

        self.offs_pos *= self.stacked
        self.offs_neg *= self.stacked

    def set_offset(self, y, offs_pos, offs_neg):
        '''
        Update two separate offset vectors for positive and negative
        values.
        '''

        y = np.array(y)

        add_pos = y.copy()
        add_pos[np.isnan(add_pos)] = 0
        add_pos[add_pos <= 0] = 0
        add_neg = y.copy()
        add_neg[np.isnan(add_neg)] = 0
        add_neg[add_neg >= 0] = 0

        print('offs_pos', offs_pos, type(offs_pos))

        print('add_pos', add_pos, type(add_pos))

        offs_pos += add_pos
        offs_neg += add_neg
        return offs_pos, offs_neg

    def get_offset(self, sgn, offs_pos, offs_neg):
        ''' Assemble positive/negative offset vector for the current
            series. '''
        offs_select = np.array([(offs_pos[i] if sgn[i] >= 0 else offs_neg[i])
                       for i in range(len(sgn))])
        return offs_select

    def get_sign(self, y):
        y = np.array(y)
        return np.sign(np.nan_to_num(y))



class StackedArea(StackedBase):
    ''' Stacked area plot. '''
    def __init__(self, *args, **kwargs):
        super(StackedArea, self).__init__(*args, **kwargs)

    def gen_single_plot_series(self, ic, ic_name, ic_color, y):
        ''' Use stacked fill_between to add areas to stacked area plot. '''

        sgn = self.get_sign(y)
        self.offs_slct = self.get_offset(sgn, self.offs_pos, self.offs_neg)


#        edgecolor = self.colors[ic_color] if self.edgecolor is None else self.edgecolor




        if self.edgewidth > 0 or (not self.edgewidth):
            edge_opacity = 0
            edgecolor = None
            plot_line = [None]
        else:
            edge_opacity = self.opacity[ic_color]
            edgecolor = self.edgecolor

            plot_line = self.ax.step(self.xpos, self.offs_slct + y,
                                    marker=None,
                                    linewidth=self.edgewidth,
                                    alpha=edge_opacity,
                                    color=self.edgecolor,
                                    where='mid',)


        plot_area = self.ax.fill_between(self.xpos, self.offs_slct, self.offs_slct + y,
                             color=self.colors[ic_color],
                             alpha=self.opacity[ic_color],
                             linewidth=0,
                             label=ic, step='mid',
                             )
#
#        from matplotlib.transforms import Affine2D
#        from matplotlib.collections import PathCollection
#
#
#        r = Affine2D().rotate_deg(90)
#
#        for x in self.ax.images + self.ax.lines + self.ax.collections:
#            trans = x.get_transform()
#            x.set_transform(r+trans)
#            if isinstance(x, PathCollection):
#                transoff = x.get_offset_transform()
#                x._transOffset = r+transoff

#        old = self.ax.axis()
#        self.ax.axis(old[2:4] + old[0:2])
#


        self.pltlgd_handles.append(mpatches.Patch(color=self.colors[ic_color],
                                                  alpha=self.opacity[ic_color]))
        self.pltlgd_labels.append(ic)

        return (plot_line[0], plot_area)

    def get_legend_handles_labels(self):
        '''
        Overwrites the parent method. Return manual legend items instead of
        from self.ax.
        '''

        return (self.pltlgd_handles, list(map(str, self.pltlgd_labels)))


class StackedGroupedBar(StackedBase):
    '''
    Grouped bars are assembled through multiple calls to StackedGroupedBar
    using the keyword argument bar=[ibar, nbars]
    '''
    def __init__(self, *args, **kwargs):
        super(StackedGroupedBar, self).__init__(*args, **kwargs)

        self.on_values = False
        self.reset_xticklabels = True

        # bars correspond to ind_axy
        self.list_bars = self.data.index.get_level_values(-1).unique().tolist()
        self.nbar = len(self.list_bars)

        # bars require widths
        if not self.barwidth:
            self.barwidth = 1 / (self.nbar + 1)

        if not self.barspace:
            self.barspace = 1. / (self.nbar * 1.1 )


    def finalize_axis(self):
        super().finalize_axis()
        self.gen_barname()

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

        for iic, ic in enumerate(self.c_list):

            for ibar, bar_name in enumerate(self.list_bars):

                self.ibar = ibar + 1 # bars are 1-indexed
                self.bar_name = bar_name

                data_bar = self.data.loc[self.data.index.get_level_values(-1) == self.bar_name]

                print(self.data)
                print(self.bar_name)

                data_bar = self.data.xs(self.bar_name, axis=0, level=-1)

                # update xpos due to bar selection
                self.generate_xpos(data_bar)

                y = np.array([iy for iy in data_bar[ic].get_values()])

                self.reset_offset(data_bar)



                print('self.xpos: ', self.ibar, self.xpos)


                ic_name = self.c_list_names[iic]
                ic_color = self.c_list_color[iic]

                self.linedict.update({ic: self.gen_single_plot_series(ic, iic, ic_color, y)})

            if self.label_format:
                self.draw_data_labels(series_name=ic_name, y=y,
                                      offs=self.offs_slct)

            self.adapt_plot_series(y)



    def gen_single_plot_series(self, ic, ic_name, ic_color, y):
        ''' Generate simple bar plot with offsets. The grouping is handled
            through the xpos.'''
        sgn = self.get_sign(y)
        self.offs_slct = self.get_offset(sgn, self.offs_pos, self.offs_neg)



        plot = self.ax.bar(self.xpos, y,
                            edgecolor=self.edgecolor,
                            color=self.colors[ic_color],
                            alpha=self.opacity[ic_color],
                            bottom=self.offs_slct,
                            align='center',
                            linewidth=self.linewidth,
                            width=self.barwidth,
                            label=ic)

        return plot[0]

    def _reset_xticklabels(self):
        '''
        For StackedGroupedBar plots the xticklabels don't include the bar names
        '''

        self.ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(len(self.data))))
        self.ax.xaxis.set_major_formatter(mticker.FixedFormatter(self.data.xs(self.data.index.get_level_values(-1)[0], level=-1).index))
        plt.setp(self.ax.get_xticklabels(), rotation=self.xticklabel_rotation)



# %
class LineBase(PlotsBase):
    ''' All we need for the subsequent definition of line-based plotting
        classes. '''
    def __init__(self, *args, **kwargs):

        PlotsBase.__init__(self, *args, **kwargs)
        self.loc_labels = 1

    def gen_single_plot_series(self, ic, y):
        '''Plot single series.'''

class LinePlot(LineBase):
    ''' Line plots. '''

    def gen_single_plot_series(self, ic, ic_name, ic_color, y):

        plot = self.ax.plot(self.xpos, y, color=self.colors[ic_color],
                     alpha=self.opacity[ic_color],
                     linewidth=self.linewidth,
                     marker=self.marker, label=ic,
                     markerfacecolor=self.markerfacecolor,
                     **self.kwargs)

        return plot[0]

class StepPlot(LineBase):
    ''' Step plots. '''
    def gen_single_plot_series(self, ic, ic_name, ic_color, y):
        plot = self.ax.step(self.xpos, y, label=ic,
                            marker=self.marker,
                            linewidth=self.linewidth,
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

    def __init__(self, data, ax, x, *args, **kwargs):

        super(BoxPlot, self).__init__(data=data, ax=ax, *args, **kwargs)

        # stacking columns
        self.data = (self.data.rename_axis('hue', axis=1)
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



def add_subplotletter(ax, n, ha='left', va='top', loc=(0.01,0.99), fs=15):
    list_abc = ['(%s)'%ii for ii in list('abcdefghijklmnopqrstuvwxyz')]

    t = ax.text(*loc, list_abc[n], transform=ax.transAxes,
            fontsize=fs, va=va, ha=ha)

    t.set_bbox(dict(facecolor='white', linewidth=0, alpha=0.5,
                    boxstyle='square,pad=0'))

# %%
if __name__ == '__main__':
    pass
#
#    df = do_winsol.data.loc[('(10, 50]', 0, 'winsol')]
#
#
#    fig, ax = plt.subplots(1,1)
#    boxplot = BoxPlot(ax=ax, x='swyr_vl', data=df)
#

#

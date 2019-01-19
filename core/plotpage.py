'''
Assembles plot pages based on the grimsel.plotting.plotting module
'''
import sys
from importlib import reload

import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


import pyAndy.core.plotting as lpplt
from  pyAndy.auxiliary import aux_sql_func as aql

from pyAndy.core.plotpagedata import PlotPageData as PlotPageData

reload(lpplt)
reload(aql)

# %%

#
#
#import numpy as np
#import matplotlib.pyplot as plt
#
## We'll use two separate gridspecs to have different margins, hspace, etc
#gs_top = plt.GridSpec(5, 2, top=0.95    )
#gs_base = plt.GridSpec(5, 1, hspace=0)
#fig = plt.figure()
#
## Top (unshared) axes
#topax = fig.add_subplot(gs_top[0,:])
#topax.plot(np.random.normal(0, 1, 1000).cumsum())
#
## The four shared axes
#ax = fig.add_subplot(gs_base[1,:]) # Need to create the first one to share...
#other_axes = [fig.add_subplot(gs_base[i,:], sharex=topax) for i in range(2, 5)]
#bottom_axes = [ax] + other_axes
#
## Hide shared x-tick labels
#for ax in bottom_axes[:-1]:
#    plt.setp(ax.get_xticklabels(), visible=False)
#
## Plot variable amounts of data to demonstrate shared axes
#for ax in bottom_axes:
#    data = np.random.normal(0, 1, np.random.randint(10, 500)).cumsum()
#    ax.plot(data)
#    ax.margins(0.05)

#%%


class PlotPage:
    """
    Sets up the page and the axes layout, making use of gridspec and subplots.
    """

    # default parameters for page layout
    dim_a4_wide = (11.69, 8.27)
    dim_a4_high = (8.27, 11.69)
    pg_layout = {'page_dim': dim_a4_high,
                 'bottom': 0.25, 'top': 0.95, 'left': 0.2, 'right': 0.8,
                 'wspace': 0.2, 'hspace': 0.2,
                 'width_ratios': None, 'height_ratios': None,
                 'axarr': None,
                 'dpi': 150}

    # scaling depending on screen resolution
    page_scale = 2#get_config('page_scale')[sys.platform]

    def __init__(self, nx, ny, sharex, sharey, **kwargs):

        self.pg_layout = PlotPage.pg_layout.copy()


        for key, val in self.pg_layout.items():
            if key in kwargs.keys():
                self.pg_layout.update({key: kwargs.pop(key)})

        page_dim = self.pg_layout.pop('page_dim')
        self.dpi = self.pg_layout.pop('dpi')
        print('dpi', self.dpi)

        axarr = self.pg_layout.pop('axarr')

        if not isinstance(axarr, np.ndarray):
            self.fig, self.axarr = plt.subplots(nrows=ny, ncols=nx,
                                                sharex=sharex,
                                                sharey=sharey,
                                                squeeze=False,
                                                gridspec_kw=self.pg_layout,
                                                dpi=self.dpi
                                                )
            self.axarr = self.axarr.T
        else:
            self.fig, self.axarr = (axarr[0][0].get_figure(), axarr)


        if not page_dim is None:
            self.fig.set_size_inches([PlotPage.page_scale * idim
                                      for idim in page_dim])
#
#    def __init__(self, nx, ny, sharex, sharey, **kwargs):
#
#        self.pg_layout = PlotPage.pg_layout.copy()
#
#        # only one of hspace or wspace can be a list!
#        self.pg_layout['hspace'] = [0, 0.1, 0, 0]
#        self.pg_layout['wspace'] = 0.2
#
#        self.pg_layout['sharex'] = [True, True, True, True, False]
#        self.pg_layout['sharey'] = False
#
#        self.pg_layout['height_ratios'] = [4,3,5,7,4]
#
#        nx = 2
#        ny = 5
#
#        hspace = self.pg_layout['hspace']
#
#        # number of gridspecs required is determined by the layout
#        # and coupling specified through hspace, wspace
#
#        # sharex, sharey can be applied at will in the corresponding loops
#
#        # how many gridspecs do we need and how many axes do they contain?
#        gs_mapy = [1]
#        for iy in range(1, ny):
#            if iy > 0 and hspace[iy] == hspace[iy - 1]:
#                gs_mapx[-1] += 1
#            elif iy > 0 and hspace[iy] != hspace[iy - 1]:
#                gs_mapx.append(1)
#
#        gs_mapx
#
#gs_mapx = [2, 3]
#
#
#
#        for key, val in self.pg_layout.items():
#            if key in kwargs.keys():
#                self.pg_layout.update({key: kwargs.pop(key)})
#
#        page_dim = self.pg_layout.pop('page_dim')
#        self.fig, self.axarr = plt.subplots(nrows=ny, ncols=nx,
#                                            sharex=sharex,
#                                            sharey=sharey,
#                                            squeeze=False,
#                                            gridspec_kw=self.pg_layout,
#                                            dpi=self.dpi
#                                            )
#        self.axarr = self.axarr.T
#
#        if not page_dim is None:
#            self.fig.set_size_inches([PlotPage.page_scale * idim
#                                      for idim in page_dim])

class PlotTiled(PlotPage):
    ''' Set up a tiled plot page. '''

    def __init__(self, pltpgdata, **kwargs):

        self.pltpgdata = pltpgdata

        # good to have a copy
        self.list_ind_pltx = self.pltpgdata.list_ind_pltx.copy()
        self.list_ind_plty = self.pltpgdata.list_ind_plty.copy()

        self.axarr = []

        defaults = {
                    'kind_def': 'LinePlot',
                    'kind_dict': {},
                    'plotkwargsdict': {},
                    'nx': len(self.list_ind_pltx),
                    'ny': len(self.list_ind_plty),
                    'sharex': False, 'sharey': False,
                    'val_axy': False,
                    'caption': True,
                    'drop_nan_cols': True,
                    'draw_now': True,
                    'legend': 'page', #'plots', 'page', 'page_plots_all', 'page_all', 'page_plots', False
                   }
        for key, val in defaults.items():
            setattr(self, key, val)
            if key in kwargs.keys():
                setattr(self, key, kwargs[key])
                kwargs.pop(key)

        # will hold all generated plots
        self.plotdict = {}

        # dict containing plot locations in the grid
        self.posdict = {}

        self.pglgd_handles = []
        self.pglgd_labels = []


        self.legend = ('none' if not self.legend
                       else ('page' if not type(self.legend) is str
                             else self.legend))

        # set up plotpage
        if self.draw_now:
            # Produce an actual figure
            kws = dict(nx=self.nx, ny=self.ny,
                       sharex=self.sharex, sharey=self.sharey,
                       **{kk: vv for kk, vv in kwargs.items()
                          if kk in PlotPage.pg_layout.keys()})
            super(PlotTiled, self).__init__(**kws)
        else:
            # Generate a dummy axarr, which is the only necessary parameter
            self.axarr = np.tile(np.nan, (self.nx, self.ny))

        # remove PlotPage kwargs from kwargs
        kwargs = {kk: vv for kk, vv in kwargs.items()
                  if not kk in PlotPage.pg_layout.keys()}

        # left-over kwargs are for plots
        self.plotkwargs = kwargs
        self.add_default_xylabel()

        # get full dictionary plot type --> columns
        self.gen_kind_dict()


        self.ax_loop()

        if self.draw_now:
            self.draw_plots()
            self.finalize_plottiled()

    @staticmethod
    def save_plot(fig, name):

        with PdfPages(name + '.pdf') as pp:
            pp.savefig(fig)

        fig.savefig(name + '.svg')
        cmd = 'inkscape --file {f}.svg --export-emf {f}.emf'.format(f=name)
        cmd = cmd.split(' ')
        subprocess.run(cmd)


    @classmethod
    def concat(cls, concat_list, concat_dir='y',
               sharex=False, sharey=False, draw_now=True, alt_align=False,
               **kwargs):

        if not concat_dir in ['x', 'y']:
            raise ValueError('Parameter concat_dir must be one of ("x", "y")')

        self = cls.__new__(cls)

        # are we concatenating along x?
        concat_x = (concat_dir == 'x')

        # get new subplots shape
        _dim_sum = 0 if concat_dir == 'x' else 1
        _dim_max = 1 if concat_dir == 'x' else 0
        len_sum = sum([obj.axarr.shape[_dim_sum] for obj in concat_list])
        len_max = max([obj.axarr.shape[_dim_max] for obj in concat_list])
        self.nx, self.ny = ((len_sum, len_max) if concat_x else
                            (len_max, len_sum))

        # generate instance new PlotPage
        if draw_now:
            kws = dict(nx=self.nx, ny=self.ny,
                       sharex=sharex, sharey=sharey,
                       **{kk: vv for kk, vv in kwargs.items()
                          if kk in PlotPage.pg_layout.keys()})
            super(PlotTiled, self).__init__(**kws)
        else:
            # Generate a dummy axarr, which is the only necessary parameter
            self.axarr = np.tile(np.nan, (self.nx, self.ny))


        # generate new plotdict dict
        self.plotdict = {}
        self.posdict = {}

        # keep track of axes which are being used
        list_ax_not_empty = []

        cnt_xy = 0 # counter along concatenation dimension
        for npo, po_slct in enumerate(concat_list):


            print(npo)
            for nx, pltx, ny, plty, plot, ax, kind in po_slct.get_plot_ax_list():

                ''''''

                if concat_x:
                    align_offset = self.ny - po_slct.ny if alt_align else 0
                else:
                    align_offset = self.nx - po_slct.nx if alt_align else 0


                gridpos_x = nx + (cnt_xy if concat_x else 0) \
                               + (align_offset if not concat_x else 0)
                gridpos_y = ny + (cnt_xy if not concat_x else 0) \
                               + (align_offset if concat_x else 0)

                print(nx, pltx, ny, plty, gridpos_x, gridpos_y)


                # raise error if plot indices are already in plotdict keys;
                # note that this is not strictly necessary but avoids trouble
                # later on (double keys in plotdict):
                if (pltx, plty) in self.plotdict.keys():
                    e = ('Trying to add a second plot {}. Make sure plot ' +
                         'indices are unique before concatenating TiledPlot ' +
                         'objects. Consider using the _name attribute as ' +
                         'an index.').format(str((pltx, plty)))
                    raise IndexError(e)

                # add plot to new plotdict
                self.plotdict[pltx, plty, kind] = plot

                # assign new axes to the original plot objects
                plot.ax = self.axarr[gridpos_x][gridpos_y]

                list_ax_not_empty.append(plot.ax)

                # define new position for the original plot objects
                self.posdict[pltx, plty] = (gridpos_x, gridpos_y)

            cnt_xy += nx + 1 if concat_x else ny + 1


        # generate list_ind_pltx/y lists
        if draw_now:
            # delete empty axes
            for ax_del in [ax for ax in self.axarr.flatten()
                           if not ax in list_ax_not_empty]:
                self.fig.delaxes(ax_del)

            for plt_slct in concat_list:
                for plot in plt_slct.plotdict.values():
                    plot.gen_plot()
                    plot.finalize_axis()

        return self

    def finalize_plottiled(self):

        if self.caption:
            self.add_caption()

        if 'page' in self.legend:
            self.add_page_legend()

    def add_default_xylabel(self):
        '''
        Modify the plot kwargs x/ylabel param if no value is provided.
        '''

        for xy in list('xy'):

            lab = '%slabel'%xy
            no_inp = (not lab in self.plotkwargs.keys() or
                      (lab in self.plotkwargs.keys()
                       and self.plotkwargs[lab] in [False, None]))

            if no_inp:
                # Setting the x/y data name by default, if available, else the
                # value name.
                axxy = getattr(self.pltpgdata, 'ind_ax%s'%xy)
                if axxy is not None:
                    self.plotkwargs.update({lab: axxy})
                elif xy != 'x':
                    self.plotkwargs.update({lab: self.pltpgdata.values})



    def add_page_legend(self, slct_plot=None, handles=None, labels=None):

        legkw = {'bbox_transform': self.fig.transFigure,
                 'bbox_to_anchor': (1, 1),
                 'fancybox': True, 'ncol': 1, 'shadow': False}

        if not handles or not labels:

            _df_lab = pd.DataFrame([c.replace('(', '').replace(')', '').split(', ')
                                    for c in self.pglgd_labels])

            for icol in _df_lab.columns:
                if len(_df_lab[icol].unique()) is 1:
                    _df_lab.drop(icol, axis=1, inplace=True)
            self.pglgd_labels = _df_lab.apply(lambda x: '(' + ', '.join(x) + ')', axis=1).tolist()

            if 'all' in self.legend or not 'page' in self.legend:
                # the second conditions means we are creating the legend
                # after the initialization of the PlotTiled object!
                handles = self.pglgd_handles
                labels = self.pglgd_labels
            else:

                if slct_plot is None:
                    _slct_plot = self.current_plot
                else:
                    _slct_plot = self.plotdict[slct_plot]

                handles = handles if handles else _slct_plot.pltlgd_handles
                labels = labels if labels else _slct_plot.pltlgd_labels

        print('Adding handles, labels %s, %s'%(str(handles), str(labels)))

        self.axarr[0][0].legend(handles, labels, **legkw)

    def gen_kind_dict(self):
        '''
        Expands the dictionary data series -> plot type.
        This dictionary is used to slice the data by plot type.
        Note: input arg kind_dict is of shape {'series_element': 'kind0', ...}
        '''

        # flatten column names
        cols_all_items = set([cc for c in self.pltpgdata.data.columns
                              for cc in c])

        # default map: columns -> default type
        kind_dict_cols = {kk: self.kind_def
                          for kk in self.pltpgdata.data.columns}

        # update using the kind_dict entries corresponding to single elements
        dct_update = {kk: vv for kk, vv in self.kind_dict.items()
                      if kk in cols_all_items
                      or kk in self.pltpgdata.data.columns}
        dct_update = {cc: [vv for kk, vv in dct_update.items() if kk in cc or kk == cc][0]
                       for cc in self.pltpgdata.data.columns
                       if any([c in cc or c == cc for c in dct_update.keys()])}
        kind_dict_cols.update(dct_update)

        # update using the kind_dict entries corresponding to specific columns
        dct_update = {kk: vv for kk, vv in self.kind_dict.items()
                      if kk in self.pltpgdata.data.columns}
        dct_update = {cc: [vv for kk, vv in dct_update.items() if kk in cc][0]
                       for cc in self.pltpgdata.data.columns
                       if any([c in cc for c in dct_update.keys()])}
        kind_dict_cols.update(dct_update)

        # invert to plot type -> data series
        kind_dict_rev = {k: [v for v in kind_dict_cols.keys()
                             if kind_dict_cols[v] == k]
                         for k in list(set(kind_dict_cols.values()))}

        print(kind_dict_rev)

        self.kind_dict_cols = kind_dict_rev




    def draw_plots(self):

        for nameplot, plot in self.plotdict.items():

#            print('Drawing %s'%str(nameplot))

            plot.draw_plot()



#            handles, labels = self.current_plot.ax.get_legend_handles_labels()
#            self.pglgd_handles += handles
#            self.pglgd_labels += labels



        # loop over plots
        #   update ax!
        #   call plot method gen_plot_series
        #   do legend stuff

    def gen_plot(self, data_slct, ipltxy, kind):
        ''' Generate plots of the selected kind. Also adds axis labels and
            the legend, if necessary. '''

        ax = self.axarr[ipltxy[0]][ipltxy[1]]

        # main call to lpplt
        self.data_slct = data_slct

        # only generate plot objects, don't draw them
        no_draw = {'draw_now': False}

        if kind in ['BoxPlot']: # requires special input kwargs
            kwargs = dict(data=data_slct, ax=ax,
                          x=self.pltpgdata.ind_axx[0], **no_draw)
            self.current_plot = lpplt.BoxPlot(**kwargs)

        # check whether plotting contains a dedicated class for this
        # plot kind; if yes, create an instance. ...
            print(lpplt.__dict__.keys())
        elif kind in lpplt.__dict__.keys():
            print(self._plotkwargs)

            kwargs = dict(data=data_slct, ax=ax, **self._plotkwargs, **no_draw)
            self.current_plot = getattr(lpplt, kind)(**kwargs)

        # ... if no, it's a pandas plot, for now
        else:
            kwargs = dict(data=data_slct, ax=ax, pd_method=kind,
                          **self._plotkwargs, **no_draw)
            self.current_plot = lpplt.PlotPandas(**kwargs)


    def get_legend_handles_labels(self, unique=True, **kwargs):

        hdls_lbls = []
        for plot in self.plotdict.values():
            plot_hdl, plot_lbl = plot.get_legend_handles_labels(**kwargs)
            hdls_lbls += list(zip(plot_hdl, plot_lbl))

        if unique:
            hdls_lbls_new = []
            for hhll in hdls_lbls:
                if not hhll[1] in [ll for _, ll in hdls_lbls_new]:
                    hdls_lbls_new.append(hhll)
            hdls_lbls = hdls_lbls_new

        return list(zip(*hdls_lbls))


    def ax_loop(self):
        '''
        Loops over
        - axis columns
        - axis rows
        - plot types as defined in self.kind_dict_cols
        Selects data for the corresponding subplot/type and calls gen_plot.
        '''
        for ipltx, slct_ipltx in enumerate(self.list_ind_pltx):
            for iplty, slct_iplty in enumerate(self.list_ind_plty):

                index_slct = tuple(([*slct_ipltx]
                                    if not self.pltpgdata.ind_pltx is None else [])
                                 + ([*slct_iplty]
                                    if not self.pltpgdata.ind_plty is None else []))


                if index_slct in self.pltpgdata.data.index or index_slct == tuple():

                    print('index_slct', index_slct)
                    data_slct_0 = pd.DataFrame(self.pltpgdata.data.loc[index_slct])
                    data_slct_0 = self.remove_nan_cols(data_slct_0)

                    # plot-specific updates of plotkwargs
                    # 1. parameters from provided plotkwargsdict
                    self._plotkwargs = self.plotkwargs.copy()
                    if index_slct in self.plotkwargsdict.keys():
                        upd = self.plotkwargsdict[index_slct]#{kk: vv for kk, vv
#                               in self.plotkwargsdict[index_slct].items()
#                               if kk in self.plotkwargs.keys()}
                        self._plotkwargs.update(upd)
                        print(upd)
                    else:
                        print('not in pkwd.')
                    # 2. title
                    if (not 'title' in self._plotkwargs.keys()) \
                        or ('title' in self._plotkwargs.keys()
                            and self._plotkwargs['title'] in [False, None]):

                        self._plotkwargs['title'] = \
                            '{}\n{}'.format(str(slct_ipltx),
                                            str(slct_iplty))
                    # 3. plotkwargs know where they are located
                    self._plotkwargs['gridpos'] = (ipltx, iplty)


                    for kind in self.kind_dict_cols.keys():
                        print(kind)
                        col_subset = [c for c in data_slct_0.columns
                                      if c in self.kind_dict_cols[kind]]

                        if col_subset:

                            print('Plotting ', index_slct, self.pltpgdata.ind_pltx,
                                                           self.pltpgdata.ind_plty,
                                                           kind)

                            data_slct = data_slct_0[col_subset]

                            _indx_drop = [ii for ii in data_slct.index.names
                                          if not ii
                                          in self.pltpgdata._ind_ax_all]

                            if _indx_drop:
                                data_slct.reset_index(_indx_drop, drop=True,
                                                      inplace=True)
                            ipltxy = [ipltx, iplty]
                            self.gen_plot(data_slct, ipltxy, kind)

                            self.plotdict[slct_ipltx, slct_iplty, kind] = self.current_plot
                            self.posdict[slct_ipltx, slct_iplty] = (ipltx, iplty)

                if str.find(self.legend, 'plots') > -1:
                    self.axarr[ipltx][iplty].legend(\
                              self.current_plot.pltlgd_handles,
                              self.current_plot.pltlgd_labels)

    def remove_nan_cols(self, df):
        '''
        Drop data columns which are nan only.

        This is relevant for PlotPageData objects resulting from
        addition.
        '''

        if self.drop_nan_cols:
            return df.loc[:, -df.isnull().all(axis=0)]
        else:
            return df

    def _gen_caption_string(self):
        '''
        Generate caption string to be added to the bottom of the plot page.
        '''

        self.caption_str = ('n_min={}, n_max={}, data_threshold={}\n' +
                            'table: {}\nfilt={}')\
                            .format(*self.pltpgdata.nsample,
                                    self.pltpgdata.data_threshold,
                                    '.'.join([str(self.pltpgdata.sc),
                                              str(self.pltpgdata.table)]),
                                    self.pltpgdata.filt)

    def add_caption(self):
        ''' Add basic information to the bottom of the page. '''
        if not 'caption_str' in self.__dict__.keys():
            self._gen_caption_string()

        plt.figtext(0.05, 0.05, self.caption_str.replace('_', ' '),
                    va='top', wrap=True)

    def get_plot_ax_list(self):
        '''
        Return all relevant plot indices and objects.

        This is useful to make specific changes to the plot after the object
        instantiation. The return list is constructed from the
        posdict and plotdict attributes. Note: The keys of the plotdict
        are (name_x, name_y, plot_kind), the keys of the posdict are
        (name_x, name_y).

        Return value:
        List of tuples (index_x, name_x, index_y, name_y,
                        plot_object, axes, plotkind) for each plot/plot_type
        '''

        return [(self.posdict[nxyk[:2]][0], nxyk[0],
                 self.posdict[nxyk[:2]][1], nxyk[1],
                 p, p.ax, nxyk[2]) for nxyk, p in self.plotdict.items()]


    @staticmethod
    def add_shared_label(text, ax1, ax2, axis='x', label_offset=0.1,
                         twinax=False, rotation=None):
        '''
        Adds an x or y-label between two axes.
        '''

        label_pos1 = (0 if axis == 'y' and not twinax else 1,
                      1 if axis == 'x' and twinax else 0)
        label_pos2 = (1 if axis == 'y' and twinax else 0,
                      0 if axis == 'x' and not twinax else 1)

        if not twinax:
            label_offset *= -1

        label_pos = (0.5 * (ax1.transAxes.transform(label_pos1)
                           + ax2.transAxes.transform(label_pos2)))
        label_pos = ax1.transAxes.inverted().transform(label_pos)
        label_pos[1 if not axis=='y' else 0] += label_offset

        if axis == 'x':
            ax1.set_xlabel(text)

            ax1.xaxis.set_label_coords(*label_pos)

            if rotation:
                ax1.yaxis.label.set_rotation(rotation)

        elif axis == 'y':
            ax1.set_ylabel(text)

            ax1.yaxis.set_label_coords(*label_pos)

            if rotation:
                ax1.yaxis.label.set_rotation(rotation)



if __name__ == '__main__':
    pass



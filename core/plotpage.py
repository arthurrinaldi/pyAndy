'''
Assembles plot pages based on the grimsel.plotting.plotting module
'''
import sys
import itertools
from importlib import reload
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pyAndy.core.plotting as lpplt
from  pyAndy.auxiliary import aux_sql_func as aql
from pyAndy.auxiliary.aux_general import get_config

reload(lpplt)
reload(aql)

class PlotPageData():

    @classmethod
    def from_df(cls, df, ind_pltx, ind_plty,
                      ind_axx, values, series, **kwargs):
        ''' Instantiate PlotPageData from DataFrame instead of PSQL table. '''

        print('kwargs', kwargs)

        do = cls(db=None, ind_pltx=ind_pltx, ind_plty=ind_plty,
                 ind_axx=ind_axx, values=values, series=series,
                 table=None, _from_sql=False, **kwargs)

        do.data_raw_0 = df.copy()
        do.update()
#
        return do

    @classmethod
    def fromdataframe(cls, df, ind_pltx, ind_plty,
                      ind_axx, values, series, **kwargs):
        ''' Instantiate PlotPageData from DataFrame instead of PSQL table. '''

        from warnings import warn
        warn('fromdataframe is deprecated... use from_df',
             DeprecationWarning, stacklevel=2)

        return cls.from_df(df, ind_pltx, ind_plty,
                           ind_axx, values, series, **kwargs)

    @property
    def data(self):
        '''
        The data attribute is implemented as property so we can
        call get_index_lists and order_data upon each assignment.
        '''
        return self.__data

    @data.setter
    def data(self, data):
        self.__data = data
        self.get_index_lists()
        self.order_data()

    def __init__(self, db='', ind_pltx=None, ind_plty=None, ind_axx=None,
                 values=None, series=None, table='', _from_sql=True, **kwargs):
        '''
        Main __init__, reads data from PSQL table by default.

        db -- SQL database
        ind_pltx -- table column for the x direction index of the plot grid
        ind_plty -- table column for the y direction index of the plot grid
        ind_axx -- table column for the x axis index
        values -- values
        series -- table column of plot series index
        table -- table name (either ['schema', 'table'] or 'schema.table')
        _from_sql -- set by class method from_df

        '''

        self._from_sql = _from_sql

        self.__data = None

        self.db = db
        self.ind_pltx = ind_pltx
        self.ind_plty = ind_plty
        self.ind_axx = ind_axx
        self.values = values
        self.series = series
        self.table = table
        self._from_sql = _from_sql

        defaults = dict(filt = [],
                        tweezer = [],
                        sc = 'public', # database schema

                        relto=None,
                        ind_rel='',
                        reltype='',

                        name=None,

                        aggfunc=np.sum,
                        data_threshold=False,
                        data_scale=False,

                        post_filt = [],
                        totals = dict(),
                        series_order = [],
                        harmonize = False,
                        padding = True,
                        replace_all = None,
                        ind_axy = None # only needed for 4D data
                        )
        for key, val in defaults.items():
            setattr(self, key, val)
            if key in kwargs.keys():
                setattr(self, key, kwargs[key])
                kwargs.pop(key)

        if self._from_sql:

            # just in case we feel like defining schema and table as 'schema.table':
            _table = self.table.split('.')
            self.sc, self.table = (_table if len(_table) == 2
                                   else [self.sc, self.table])

            self.update()

    def update(self):

        if self._from_sql:
            print('Getting data from sql table.')
            self.get_data_raw()
        else:
            print('Getting data from DataFrame.')
        self._prep_data()
        self.get_index_lists()


    def harmonize_indices(self, df):
        '''
        Expand the DataFrame df in such a way that all combinations of the
        values in the index cols are present. Then add value column(s).
        '''

        cols = list(set(self._index + self._series))
        values = self._values

        na = float('nan')
        fillval = na if type(self.harmonize) is bool else self.harmonize

        for ncol, icol in enumerate(cols):
            if ncol == 0:
                df_1 = pd.DataFrame(df[icol].unique().tolist(),
                                    columns=[icol])
                df_1['key'] = 1
            else:
                df_add = pd.DataFrame(df[icol].unique().tolist(),
                                      columns=[icol])
                df_add['key'] = 1  # key is always 1 so merge yields all combs

                df_1 = pd.merge(df_1, df_add, on='key')


        # the following conversion is a work-around due to a pandas bug
        # https://github.com/pandas-dev/pandas/pull/21310

        converted_bool_cols = []
        for _df in [df_1, df]:
            for col in _df.columns:
                if _df[col].dtype.name == 'bool':
                    _df[col] = _df[col].astype(str)
                    converted_bool_cols.append(col)

        df_1 = df_1[cols].join(df.set_index(cols)[values], on=cols).fillna(fillval)

#        for col in converted_bool_cols:
#                df_1[col] = df_1[col].astype(bool)

        return df_1


    def get_data_raw(self):

        self.data_raw_0 = aql.read_sql(self.db, self.sc, self.table,
                                       filt=self.filt, tweezer=self.tweezer,
                                       verbose=False)

        self.data_raw_0 = self.data_raw_0.drop_duplicates()

    def print_counts_unique_values(self):
        '''
        Useful to identify which data is being aggregated.
        '''

        # display unique value in input table
        df = self.data_raw_0
        for ic in df.columns:
            lst_val = df[ic].drop_duplicates().tolist()
            if len(lst_val) < 0.3 * len(df):
                print(ic, len(lst_val))


    def _prep_data(self):
        '''
        Read data from the table defined by the keyword argument table.
        Returns a pivot table with layout
        - row indices: ind_pltx, ind_plty, ind_axx
        - columns: series
        '''

        self.print_counts_unique_values()

        # Series as is.
        # Is it still possible to have multiple values as series??
        # How to handle the self._index + self._series below??
        # Could self._series become [] if series is none??
        self._series = self.series #if not series in index else None

        for iind in ['pltx', 'plty', 'axx', 'axy']:
            if not getattr(self, 'ind_' + iind):
                setattr(self, 'ind_' + iind, None)

        # Best way to make sure these are empty lists?
        self._ind_pltx = self.ind_pltx if not self.ind_pltx is None else []
        self._ind_plty = self.ind_plty if not self.ind_plty is None else []
        self._ind_axx = self.ind_axx if not self.ind_axx is None else []
        self._ind_axy = self.ind_axy if not self.ind_axy is None else []

        # assemble total index
        self._index = [*self._ind_pltx, *self._ind_plty,
                       *self._ind_axx, *self._ind_axy]
        self._ind_ax_all = [*self._ind_axx, *self._ind_axy]

        # make sure rel attributes are lists
        for iatt in ['relto', 'reltype', 'ind_rel']:
            _ind = getattr(self, iatt)
            _ind_lst = _ind if type(_ind) is list else [_ind]
            setattr(self, iatt + '_lst', _ind_lst)

        # make sure post_filt is a list of lists
        post_filt_lst = [f if type(f) is list else [f]
                         for f in self.post_filt]

        # ind_rel must only be added if relevant + not there yet
        for _ind_rel in self.ind_rel_lst:
            no_add_ind_rel = (_ind_rel in self._index
                              or _ind_rel in self._series
                              or _ind_rel is False)
            _ind_rel = [[_ind_rel] if not no_add_ind_rel else []][0]
            self._index += _ind_rel

        self._index = [c for c in self._index if not c == '']

        # if '_values' is in the index or the series index,
        # modify the data_raw_0 accordingly
        if '_values' in self._index or '_values' in self._series:

            self.data_raw_0.reset_index(drop=True, inplace=True)
            self.data_raw_0.index.names = ['index']

            # create value name column by stacking
            df1 = pd.DataFrame(self.data_raw_0[self.values]
                                   .stack().rename('/'.join(self.values)))
            df1.index.names = ['index', '_values']

            # get residual data_raw_0 without value columns
            df2 = self.data_raw_0[[c for c in self.data_raw_0.columns
                                   if not c in self.values]]

            # join residuals df2 to expanded df1
            df_new = df1.reset_index().join(df2, on=df2.index.names)
            df_new = df_new.drop('index', axis=1)

            self.data_raw_0 = df_new

            self._values = ['/'.join(self.values)]

        else:
            self._values = self.values

        # add _name column if provided. This is convenient for
        # plotpagedata addition
        if self.name is not None:
            self.data_raw_0['_name'] = self.name

        # set pivot keyword arguments for various applications
        self._pv_kws = {'values': self._values,
                        'index': self._index,
                        'columns': self._series}

        # range of sample size for reporting
        self.nsample = self.calc_minmax_sample_size()

        # Now that we have the index, call method to fill numerical index
        # values with left padded zeros
        if self.padding:
            self.fill_numerical_indices()

        # make sure the selected columns are complete (if desired)
        self.data_raw = (self.harmonize_indices(self.data_raw_0)
                         if self.harmonize
                         or (type(self.harmonize) is not bool)
                         else self.data_raw_0.copy())

        # calculate values relative to reference, if relevant
        for ind_rel, relto, reltype in zip(self.ind_rel_lst, self.relto_lst, self.reltype_lst):
            bool_calc_rel = relto != None and (True if ind_rel else False)
            if bool_calc_rel:
                print(ind_rel, relto, reltype )
                self.data_raw = self.calc_relative_raw_multi(self.data_raw,
                                                             ind_rel,
                                                             relto, reltype)

        if self.post_filt and self.ind_rel:
            for ind_rel, val_lst in zip(self.ind_rel_lst, post_filt_lst):
                if len(val_lst) > 0:
                    self.post_filtering_raw(ind_rel, val_lst)


        # make sure all index values are strings or int or float:
        for col in self._pv_kws['columns'] + self._pv_kws['index']:
            if isinstance(self.data_raw[col].dtype, np.dtype):
                self.data_raw[col] = self.data_raw[col].astype(str)
            

        # calculate pivot table
        dfpv = self.data_raw.pivot_table(**self._pv_kws, aggfunc=self.aggfunc)
        dfpv = pd.DataFrame(dfpv)

        self.__data = dfpv.copy()

        # make tuples out of column names
        self.__data.columns = [tuple([c]) if type(c) != tuple else c
                             for c in self.__data.columns]

        # apply scaling factors to __data, if applicable
        self.__data = self.scale_data(self.__data)

        # shouldn't this happen before calc_relative?
        self.calc_totals()  # calculate sums of selected series

#        bool_calc_rel = ((True if self.relto else False)
#                          and (True if self.ind_rel else False))
#        if bool_calc_rel:
#            self.__data = self.calc_relative(self.__data)

        # set values below a certain threshold equal zero
        self.__data = self.cut_threshold(self.__data)

        # sort series according to external list
        self.order_data()


    def fill_numerical_indices(self):
        '''
        This fills numerical indices with an appropriate number of left
        padded zeros in order to preserve the right order when sorting them.
        '''

        for iind in self._index + self.series:
            first = self.data_raw_0[iind].iloc[0]

            is_numstr = ((type(first) is str)
                         and (first[0].isnumeric()))

            if is_numstr:
                # get maximum length of relevant numerical index
                fill_len = max(list(map(len, self.data_raw_0[iind].unique())))

                fill_zeros = lambda x: x.zfill(fill_len) if not x==None else x
                self.data_raw_0[iind] = self.data_raw_0[iind].apply(fill_zeros)

    def calc_minmax_sample_size(self):
        '''
        Minimum and maximum sample size at each data point for reporting
        this helps to identify inappropriately filtered data.
        '''
        self._dfpv_count = self.data_raw_0.pivot_table(**self._pv_kws, aggfunc=len)
        return [self._dfpv_count.min().min(), self._dfpv_count.max().max()]

    def post_filtering_raw(self, ind_rel, val_lst):

        self.data_raw = self.data_raw.loc[self.data_raw[ind_rel].isin(val_lst)]

    def post_filtering(self):
        ''' Filters dataframe for certain values of ind_rel '''
        self.__data = self.__data.loc[self.__data.index
                                           .get_level_values(self.ind_rel)
                                           .isin(self.post_filt)]
#        if len(self.post_filt) < 2:
##            self.__data = self.__data.reset_index(self.ind_rel, drop=True)
#        else:
#            index = [c for c in self.__data.index.names
#                     if not c in self.values + [self.ind_rel]]
#            self.__data = (self.__data.reset_index()
#                             .pivot_table(columns=self.ind_rel,
#                                          index=index, values=self.values))

    def cut_threshold(self, dfpv):
        '''
        For when small just equals zero.
        '''

        # sometimes small equals zero
        if self.data_threshold:
            dfpv = dfpv.applymap(lambda x: 0
                                 if abs(x) < self.data_threshold else x)
        return dfpv


    def calc_relative_raw_multi(self, df, ind_rel, relto, reltype):



        can_do = relto in df[ind_rel].unique().tolist()

        if can_do:

            clst = self._index + self._series
            clst = [c for c in clst if not c in ind_rel]

            df_rel = df.pivot_table(index=clst, values=self._values,
                                    columns=ind_rel, aggfunc=np.sum)
            df_rel['ref'] = df_rel[tuple(self._values + [relto])]

            for ic in df_rel.columns:
                print(ic)
                if reltype in ['ratio', 'share']:
                    df_rel[ic] /= df_rel['ref']
                    if reltype == 'share':
                        df_rel[ic] -= 1
                elif reltype == 'absolute':
                    df_rel[ic] -= df_rel['ref']

            df_rel = df_rel.drop('ref', axis=1)

            return df_rel.stack().reset_index()

        else:
            raise ValueError(('calc_relative: The reference value {} ' +
                   'does not exist in ind_rel.').format(self.relto))



    def calc_relative_raw(self, df):

        can_do = self.relto in df[self.ind_rel].unique().tolist()

        if can_do:

            clst = self._index + self._series
            clst = [c for c in clst if not c in self.ind_rel]

            df_rel = df.pivot_table(index=clst, values=self._values,
                                    columns=self.ind_rel, aggfunc=np.sum)
            df_rel['ref'] = df_rel[tuple(self._values + [self.relto])]

            for ic in df_rel.columns:
                if self.reltype in ['ratio', 'share']:
                    df_rel[ic] /= df_rel['ref']

                    if self.reltype == 'share':
                        df_rel[ic] -= 1


                elif self.reltype == 'absolute':
                    df_rel[ic] -= df_rel['ref']
            df_rel = df_rel.drop('ref', axis=1)

            return df_rel.stack().reset_index()

        else:
            print('''calc_relative: The reference value {}
                     does not exist in ind_rel.'''.format(self.relto))
            raise ValueError


    def calc_relative(self, dfpv):

        if self.relto in dfpv.index.get_level_values(self.ind_rel):

            columns_0 = dfpv.columns
            dfpv.columns = [''.join(list(map(str, c)))
                            for c in dfpv.columns]

            dfpv_ref = dfpv.loc[dfpv.index.get_level_values(self.ind_rel)
                                == self.relto]

            dfpv_ref.columns = [c + '_ref' for c in dfpv_ref.columns]
            dfpv_ref = dfpv_ref.reset_index(self.ind_rel, drop=True)

            dfpv_1 = (dfpv.reset_index()
                          .join(dfpv_ref,
                                on=[ii for ii in dfpv_ref.index.names
                                    if not ii == self.ind_rel])
                          .set_index(dfpv.index.names))

            for icol in dfpv.columns:
                if self.reltype == 'ratio':
                    dfpv_1[icol] /= dfpv_1[icol + '_ref']
                elif self.reltype == 'share':
                    dfpv_1[icol] = (- 1 + dfpv_1[icol]
                                    / dfpv_1[icol + '_ref'])
                elif self.reltype == 'absolute':
                    dfpv_1[icol] = (dfpv_1[icol] - dfpv_1[icol + '_ref'])

            dfpv = dfpv_1[[cc for cc in dfpv_1.columns
                           if str.find(cc, '_ref') == -1]]

            dfpv = dfpv.replace(-np.inf, np.nan).replace(np.inf, np.nan)
            dfpv.columns = columns_0

            dfpv = self.cut_threshold(dfpv)

            return dfpv

        else:
            raise ValueError(('calc_relative: The reference value %s '
                     + 'does not exist in ind_rel.') %self.relto)

    def calc_totals(self):
        ''' Calculate sums of selected data series. '''

        # special case: all columns
        for k, v in self.totals.items():
            if v == ['all']:
                self.totals.update({k: [c[-1] for c in self.__data.columns]})

        if len(self.totals) > 0:
            for totcol, sumcols in self.totals.items():
                nc = len(self.__data.columns[0])
                newcol = tuple([''] * (nc - 1) + [totcol])
                self.__data[newcol] = self.__data[[c for c
                                                   in self.__data if c[-1]
                                                   in sumcols]].sum(axis=1)

    def _cols_to_list(self, ind, none_val=[(0,)]):
        ''' Returns a list of tuples of columns and or row indices. '''
        if not ind is None:
            return (self.__data.reset_index()[ind].drop_duplicates()
                             .apply(lambda x: tuple(x), axis=1).tolist())
        else:
            return none_val

    def get_index_lists(self):
        ''' Get lists for all relevant index values. '''
        for iind in ['ind_pltx', 'ind_plty', 'ind_axx', 'ind_axy']:
            ind_list = getattr(self, iind)
            setattr(self, 'list_' + iind, self._cols_to_list(ind_list))
        self.list_ind_axy = self._cols_to_list(self.ind_axy, None)

        self.list_series = self.__data.columns.unique().tolist()

    def scale_data_raw(self, df):

        for kk in self.data_scale.keys():
            for vv, rows in self.data_scale[kk].items():
                df.loc[df[kk].isin(self.data_scale[kk][vv]), self._values]

    def scale_data(self, df):
        '''
        self.data_scale can be a float or a dictionary with keys from the
        lowest column level
        '''

        if self.data_scale:
            list_cols = [c[-1] for c in df.columns]
            is_dict = type(self.data_scale) == dict
            if is_dict:
                data_scale_all = {col: self.data_scale[col]
                                  if (col in self.data_scale.keys())
                                  else 1 for col in list_cols}
            else:
                data_scale_all = {col: self.data_scale for col in list_cols}

            for slct_col in [c for c in df.columns]:
                df[slct_col] *= data_scale_all[slct_col[-1]]

        return df

    def order_data(self):
        ''' order lowest column level by list self.series_order '''
        if len(self.series_order) > 0:
            list_ord = [[idat for idat in self.__data.columns
                         if idat[-1] == iord]
                        for iord in self.series_order]
            list_ord = list(itertools.chain(*list_ord))

            list_rest = [c for c in self.__data.columns if not c in list_ord]

            if len(list_rest) > 0:
                data_rest = self.__data[list_rest]
            else:
                data_rest = pd.DataFrame()
            self.__data = pd.concat([self.__data[list_ord], data_rest], axis=1)


    def replace_characters(self):

        dict_char = {'_': '-'}

        for chold, chnew in dict_char.items():
            # replace in columns:
            self.__data.columns = [tuple([str(cc).replace(chold, chnew)
                                        for cc in c])
                                 for c in self.__data.columns]

        self.get_index_lists()

    def get_plot_indices(self, select=[]):
        '''
        Returns index values corresponding to plots, i.e. w/out the axes.
        '''

        select_levels = [ii for ii, nn in enumerate(self.__data.index.names)
                         if not nn in self._ind_axx + self._ind_axy]

        return set([tuple(ii[i] for i in select_levels)
                    for ii in self.__data.index.values
                    if any([ss in ii for ss in select])])

    def copy(self):

        return copy.copy(self)

    def __repr__(self):

        ret = ('<%s.%s object at %s>' % (self.__class__.__module__,
                                         self.__class__.__name__,
                                         hex(id(self)))
             + '\ndata: ' + str(self.__data.head())
             + '\nind_pltx: %s' %self.ind_pltx
             + '\nind_plty: %s' %self.ind_plty
             + '\nind_axx: %s'%self.ind_axx
             + '\nind_axy: %s'%self.ind_axy
             + '\nseries: %s'%self.series
             )

        return ret

    def __add__(self, other):

        # make sure the indices are compatible
        if not self.__data.index.names == other.__data.index.names:
            raise ValueError(
                'Indices of PlotPageData objects to be added don\'t match.')

        # make sure all columns have the same number of elements
        if not (# all column name lengths of each must be equal
                all([len(cc) == len(_do.__data.columns[0])
                     for _do in [self, other]
                     for cc in _do.__data.columns])
                # length of first elements of the two must be equal
                and len(self.__data.columns[0]) == len(other.__data.columns[0])):
            raise ValueError(
                'Inconsistent length of series elements names.')

        do_add = self.copy()

        # joining shaped data; sort must be False, otherwise the manually
        # defined series_order will be gone
        do_add.__data = pd.concat([self.__data, other.__data], sort=False)

        do_add.data_raw = pd.concat([self.data_raw, other.data_raw],
                                    axis=0, sort=True)

        do_add.__data.columns = [tuple([c]) if type(c) != tuple else c
                               for c in do_add.__data.columns]

        do_add.get_index_lists()


        return do_add


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

        if axarr is None:
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

        plt.figtext(0.05, 0.05, self.caption_str, va='top', wrap=True)

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



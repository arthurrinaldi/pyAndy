#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 13:29:24 2019

@author: user
"""

import numpy as np
import pandas as pd
import copy
import itertools

import inspect
import wrapt

import pyAndy.auxiliary.aux_sql_func as aql


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
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.get_index_lists()
        self.order_data()


    @wrapt.decorator
    def _attributes_to_lists(f, self, args, kwargs):
        ''' Decorator to convert attributes to lists if they are not. '''

        print('~'*90)
        print(args, kwargs)
        print('~'*90)

        arg_names = inspect.getfullargspec(f).args[1:] # first is self
        kwargs_args = dict(zip(arg_names, args))

        kwargs.update(kwargs_args)

        list_kws = ['ipx', 'ipy',
                    'ind_pltx', 'ind_plty', 'ind_axx', 'values',
                    'series', 'ind_axy']

        for key, val in kwargs.items():
            if key in list_kws:
                if not type(val) in (tuple, list):
                    kwargs[key] = [val] if val else []
                elif isinstance(val, tuple):
                    kwargs[key] = list(val)

        print('+'*90)
        print(kwargs)
        print('+'*90)

        return f(**kwargs)


    @_attributes_to_lists
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

        self._data = None

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
                        ind_rel=None,
                        reltype=None,

                        drop_nan_cols=True,
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

        self._init_indices()

        if self._from_sql:
            print('Getting data from sql table %s.'%self.table)
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


        keep = self._ind_ax_all + self._index + self.ind_rel_lst + self.series + self.values
        keep = {ind for ind in keep if not ind in [None, '', '_name']}

        self.data_raw_0 = aql.read_sql(self.db, self.sc, self.table,
                                       filt=self.filt, tweezer=self.tweezer,
                                       verbose=True, keep=keep)

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

    def _init_indices(self):

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



    def _prep_data(self):
        '''
        Read data from the table defined by the keyword argument table.
        Returns a pivot table with layout
        - row indices: ind_pltx, ind_plty, ind_axx
        - columns: series
        '''

        self.print_counts_unique_values()


        try:
            self.data_raw_0.loc[:, self.values] = self.data_raw_0.loc[:, self.values].astype(float)
        except:
            raise ValueError('Value columns %s could '%self.values
                              + 'not be cast as float')

        # make sure post_filt is a list of lists
        post_filt_lst = (self.post_filt
                         if len(self.post_filt) > 0
                             and isinstance(self.post_filt[0], list)
                         else [self.post_filt])

        # ind_rel must only be added if relevant + not there yet
        for _ind_rel in self.ind_rel_lst:
            no_add_ind_rel = (_ind_rel in self._index
                              or _ind_rel in self._series
                              or _ind_rel is False)
            _ind_rel = [[_ind_rel] if not no_add_ind_rel else []][0]
            self._index += _ind_rel

        self._index = [c for c in self._index if not c in ('', None)]

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
        for col in self._pv_kws['columns']:# + self._pv_kws['index']:
            if isinstance(self.data_raw[col].dtype, np.dtype):
                self.data_raw[col] = self.data_raw[col].astype(str)


        # calculate pivot table
        dfpv = self.data_raw.pivot_table(**self._pv_kws, aggfunc=self.aggfunc)
        dfpv = pd.DataFrame(dfpv)

        self._data = dfpv.copy()

        # make tuples out of column names
        self._data.columns = [tuple([c]) if type(c) != tuple else c
                             for c in self._data.columns]

        # apply scaling factors to _data, if applicable
        self._data = self.scale_data(self._data)

        # shouldn't this happen before calc_relative?
        self.calc_totals()  # calculate sums of selected series

#        bool_calc_rel = ((True if self.relto else False)
#                          and (True if self.ind_rel else False))
#        if bool_calc_rel:
#            self._data = self.calc_relative(self._data)

        # set values below a certain threshold equal zero
        self._data = self.cut_threshold(self._data)

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
        self._data = self._data.loc[self._data.index
                                           .get_level_values(self.ind_rel)
                                           .isin(self.post_filt)]
#        if len(self.post_filt) < 2:
##            self._data = self._data.reset_index(self.ind_rel, drop=True)
#        else:
#            index = [c for c in self._data.index.names
#                     if not c in self.values + [self.ind_rel]]
#            self._data = (self._data.reset_index()
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


        list_ind_rel = df[ind_rel].unique().tolist()

        can_do = relto in list_ind_rel

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


    @_attributes_to_lists
    def _merge_plt_indices(self, ipx, ipy):

        index_slct = itertools.chain.from_iterable((ipx, ipy))

        ind_mrg = tuple(ii for ii in index_slct
                        if not ii is None)

        return ind_mrg

    @_attributes_to_lists
    def _get_data_by_index(self, ipx=(None,), ipy=(None,)):
        ''' Return extracted data for a given index combination. '''

        assert (len(ipx) is len(self.ind_pltx)) if self.ind_pltx else (len(ipx) == 1 and ipx[0] == None)
        assert (len(ipy) is len(self.ind_plty)) if self.ind_plty else (len(ipy) == 1 and ipy[0] == None)

        return self.data.loc[self._merge_plt_indices(ipx, ipy)]

    def __getitem__(self, ind):
        ''' Returns slice of self.data based on index selection. '''

        ipx, ipy = ind


        if not self._merge_plt_indices(ipx, ipy) in self.data.index:

            raise KeyError('ipx = %s, ipy = %s is not in the index'%(ipx, ipy))

        return self._get_data_by_index(*ind)


    def remove_nan_cols(self, df):
        '''
        Drop data columns which are nan only.

        This is relevant for PlotPageData objects resulting from
        addition.
        '''

        if self.drop_nan_cols:
            return df.loc[:, ~df.isnull().all(axis=0)]
        else:
            return df


    def get_data(self):

        self._iter_ind = itertools.product(enumerate(self.list_ind_pltx),
                                           enumerate(self.list_ind_plty))

        def flatten(ind): return tuple(itertools.chain.from_iterable(ind))
        self._iter_ind = list(ind for ind in map(flatten, self._iter_ind))


        data_list = []

        for npx, ipx, npy, ipy in self._iter_ind:
            try:
                data_list.append(tuple([npx, ipx, npy, ipy,
                                        self.remove_nan_cols(self[ipx, ipy])]))
            except KeyError: continue

        return data_list



    def __iter__(self):
        '''
        Sets up combinations of plot indices.
        '''

        self._iter_ind = itertools.product(enumerate(self.list_ind_pltx),
                                           enumerate(self.list_ind_plty))

        def flatten(ind): return tuple(itertools.chain.from_iterable(ind))
        self._iter_ind = (ind for ind in map(flatten, self._iter_ind))

        return self

    def __next__(self):

        npx, ipx, npy, ipy = self._iter_ind.__next__()

        try:
            return tuple([npx, ipx, npy, ipy, self[ipx, ipy]])

        except KeyError:
            self.__next__()


    def get_plot_nxy(self):
        ''' Return size of plot array. '''

        return (len(self.list_ind_pltx),
                len(self.list_ind_plty))

    def calc_totals(self):
        ''' Calculate sums of selected data series. '''

        # special case: all columns
        for k, v in self.totals.items():
            if v == ['all']:
                self.totals.update({k: [c[-1] for c in self._data.columns]})

        if len(self.totals) > 0:
            for totcol, sumcols in self.totals.items():
                nc = len(self._data.columns[0])
                newcol = tuple([''] * (nc - 1) + [totcol])
                self._data[newcol] = self._data[[c for c
                                                   in self._data if c[-1]
                                                   in sumcols]].sum(axis=1)

    def _cols_to_list(self, ind, none_val=[(None,)]):
        ''' Returns a list of tuples of columns and or row indices. '''
        if not ind is None:
            return (self._data.reset_index()[ind].drop_duplicates()
                             .apply(lambda x: tuple(x), axis=1).tolist())
        else:
            return none_val

    def get_index_lists(self):
        ''' Get lists for all relevant index values. '''
        for iind in ['ind_pltx', 'ind_plty', 'ind_axx', 'ind_axy']:
            ind_list = getattr(self, iind)
            setattr(self, 'list_' + iind, self._cols_to_list(ind_list))
        self.list_ind_axy = self._cols_to_list(self.ind_axy, None)

        self.list_series = self._data.columns.unique().tolist()

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
        if self.series_order:
            list_ord = [[idat for idat in self._data.columns
                         if idat[-1] == iord]
                        for iord in self.series_order]
            list_ord = list(itertools.chain(*list_ord))

            list_rest = [c for c in self._data.columns if not c in list_ord]

            if list_rest:
                data_rest = self._data[list_rest]
            else:
                data_rest = pd.DataFrame()
            self._data = pd.concat([self._data[list_ord], data_rest], axis=1)


    def replace_characters(self):

        dict_char = {'_': '-'}

        for chold, chnew in dict_char.items():
            # replace in columns:
            self._data.columns = [tuple([str(cc).replace(chold, chnew)
                                        for cc in c])
                                 for c in self._data.columns]

        self.get_index_lists()

    def get_plot_indices(self, select=[]):
        '''
        Returns index values corresponding to plots, i.e. w/out the axes.
        '''

        select_levels = [ii for ii, nn in enumerate(self._data.index.names)
                         if not nn in self._ind_axx + self._ind_axy]

        return set([tuple(ii[i] for i in select_levels)
                    for ii in self._data.index.values
                    if any([ss in ii for ss in select])])

    def copy(self):

        return copy.copy(self)

    def __repr__(self):

        ret = ('<%s.%s object at %s>' % (self.__class__.__module__,
                                         self.__class__.__name__,
                                         hex(id(self)))
             + '\ndata: ' + str(self._data.head())
             + '\nind_pltx: %s' %self.ind_pltx
             + '\nind_plty: %s' %self.ind_plty
             + '\nind_axx: %s'%self.ind_axx
             + '\nind_axy: %s'%self.ind_axy
             + '\nseries: %s'%self.series
             )

        return ret

    def __add__(self, other):

        # make sure the indices are compatible
        if not self._data.index.names == other._data.index.names:
            raise ValueError(
                'Indices of PlotPageData objects to be added don\'t match.')

        # make sure all columns have the same number of elements
        if not (# all column name lengths of each must be equal
                all([len(cc) == len(_do._data.columns[0])
                     for _do in [self, other]
                     for cc in _do._data.columns])
                # length of first elements of the two must be equal
                and len(self._data.columns[0]) == len(other._data.columns[0])):
            raise ValueError(
                'Inconsistent length of series elements names.')

        do_add = self.copy()

        # joining shaped data; sort must be False, otherwise the manually
        # defined series_order will be gone
        do_add._data = pd.concat([self._data, other._data], sort=False)

        do_add.data_raw = pd.concat([self.data_raw, other.data_raw],
                                    axis=0, sort=True)

        do_add._data.columns = [tuple([c]) if type(c) != tuple else c
                               for c in do_add._data.columns]

        do_add.get_index_lists()


        return do_add




if __name__ == '__main__':

# %%

    import grimsel.auxiliary.maps as maps

    sc_out = 'out_levels'
    slct_nd = 'DE0'
    db = 'storage2'

    mps = maps.Maps(sc_out, db)

    ind_pltx = ['sta_mod']
    ind_plty = ['pwrerg_cat']
    ind_axx = ['sy']
    values = ['value_posneg']

    series = ['bool_out', 'fl']
    table = sc_out + '.analysis_time_series'


    stats_data = {'DE0': '%agora%',
                  'FR0': '%eco2%',
                  'CH0': '%entsoe%'}

    filt = [
            ('nd', [slct_nd]),
            ('swfy_vl', ['yr2015', 'nan'], ' LIKE '),
#            ('fl', ['%nuclear%'], ' LIKE '),
#            ('swchp_vl', ['chp_off']),
#            ('swcadj_vl', ['adjs']),
            ('run_id', [0, -1]),
            ('wk_id', [5]),
            ('sta_mod', ['%model%', stats_data[slct_nd]], ' LIKE '),
            ('sta_mod', ['%model%'], ' LIKE '),
            ('pwrerg_cat', ['%pwr%'], ' LIKE '),
            ('fl', ['dmnd', '%coal%', '%nuc%', '%lig%', '%gas', 'load_prediction_d',
                    'wind_%', '%photo%', '%bio%', 'lost%', 'dmnd_flex'], ' LIKE ')
            ]
    post_filt = [] # from ind_rel

    lst_series = aql.read_sql(db, sc_out, 'def_pp_type')['pt'].tolist()

    dict_series_order = {'BAL': -100,
                         'WAS': -50,
                         'NUC': -75,
                         'LIG': 10,
                         'natural_gas': 40,
                         'reservoir': 2000,
                         'export': -200,
                         'import': 1000,
                         'run_of_river': -10
                         }

    df_series_order = aql.read_sql(db, sc_out, 'def_plant', keep=['pp', 'pt_id', 'nd_id', 'fl_id'])
    df_series_order = df_series_order.join(aql.read_sql(db, sc_out, 'def_pp_type').set_index('pt_id')['pt'], on='pt_id')
    df_series_order = df_series_order.join(aql.read_sql(db, sc_out, 'def_fuel').set_index('fl_id')['fl'], on='fl_id')
    df_series_order = df_series_order.join(aql.read_sql(db, sc_out, 'def_node').set_index('nd_id')['nd'], on='nd_id')
    df_series_order['pp_red'] = df_series_order.pp.apply(lambda x: x.split('_')[1])

    df_series_order = df_series_order.loc[df_series_order.nd.isin([f for f in filt if 'nd' == f[0]][0][1])]

    df_series_order['rank'] = np.inf

    for icol in ['pp_red', 'pp_red', 'pt']:
        df_order_dict = pd.DataFrame.from_dict(dict_series_order, columns=['rank_new'], orient='index')
        df_order_dict.index.names = [icol]

        df_series_order = df_series_order.join(df_order_dict, on=df_order_dict.index.names).fillna(1e10)

        df_series_order['rank'] = df_series_order[['rank', 'rank_new']].min(axis=1)
        df_series_order = df_series_order.drop('rank_new', axis=1)

    series_order = df_series_order.sort_values(['rank', 'pp']).fl.unique().tolist()

    data_kw = {'filt': filt, 'post_filt': post_filt, 'data_scale': {'dmnd': -1},
               'totals': {'others': ['waste_mix']},
               'data_threshold': 1e-9, 'aggfunc': np.sum, 'harmonize': False,
               'series_order': series_order}

    do = PlotPageData(db, ind_pltx, ind_plty, ind_axx, values, series,
                            table, **data_kw)

    do.data = do.data.fillna(0).applymap(float)

    self = do

# %%
    # delete data aggregated in others
    do.data = do.data.loc[:, ~do.data.columns.isin(do.totals['others'])]




# %%
























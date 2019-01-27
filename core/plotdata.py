#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:52:44 2019

@author: user
"""
import itertools
import numpy as np


'''
add test

    x = np.arange(2010, 2021, 1)

    data = pd.DataFrame(dict(year=x,
                      sin=np.sin(x),
                      const=np.ones(np.size(x)))).set_index('year')

    --> get_data_offset
'''



class PlotData():
    '''
    This class holds the data for the PlotsBase class.

    Provides convenience methods, especially related to
    stacked data offset management.
    '''

    def __init__(self, data, stacked, on_values):

        self.on_values = on_values

        self.data = data

        self.c_list, self.c_list_names, self.c_list_color = self.get_c_list()

        self.data_offset = self.get_data_offset() if stacked else self.data * 0

        self.xpos = None

    @property
    def data(self):
        '''
        Remove index from default data if on_Values is False.

        For the PlotPandas class the data table is used *as is*.
        It requires an alternative table if on_values is False,
        the xpos values are not the original ones.
        '''

        return (self._data if self.on_values
                else self._data.reset_index(drop=True))

    @data.setter
    def data(self, data):
        self._data = data




    def get_data_offset(self):
        '''
        Returns data offset for each data point.

        Depending on whether the data point is positive or negative,
        the cumulative positive or negative offset is selected. That way,
        positive values add to the positive stack and negative values to the
        negative stack.

        Returns:
            DataFrame: the data offset table
        '''

        data_pos = self.data.where(self.data > 0).shift(1, axis=1).fillna(0).cumsum(axis=1)
        data_neg = self.data.where(self.data < 0).shift(1, axis=1).fillna(0).cumsum(axis=1)

        return (data_pos[self.data > 0].fillna(0)
                + data_neg[self.data < 0].fillna(0))


    def init_xpos(self, data=False, barspace=None, ibar=1, nbar=1,
                 x_offset=False, on_values=None):
        '''
        List-like object for x-axis positions.

        Either directly from data or generic, depending on the argument
        on_values.
        '''
        if isinstance(data, bool):
            data = self.data

        if not isinstance(on_values, bool):
            on_values = self.on_values


        if on_values:
            xpos = data.index.get_values().tolist()
        else:
            barspace = 1 if not barspace else barspace

            xpos = (np.arange(len(data))
                    + x_offset
                    + (ibar - 0.5 * (nbar - 1) - 1)
                    * barspace)

        self.xpos = xpos

        return xpos


    def get_c_list(self):
        '''
        Return series list, based of data columns.

        Data series are expected to be organized in columns. c_list is
        the list of columns.
        '''
        c_list = [c for c in self.data.columns]

        # c_list_names are used for indexing (colors etc). In case of
        # multiindex columns only the last element is selected.
        c_list_names = c_list.copy()
        if type(c_list[0]) in [list, tuple]:
            c_list_color = [cc[-1] for cc in c_list]

            # get relevant dimensions
            dims = []
            for idim in range(len(c_list[0])):
                if len(set([c[idim] for c in c_list])) > 1:
                    dims.append(idim)
            c_list_names = [list(c) for c in
                            (np.array(c_list).T[dims].T)]
        else:
            c_list_color = c_list
            c_list_names = c_list

        return c_list, c_list_names, c_list_color


    def get_series_y(self, ic):

        return np.array([iy for iy in self.data[ic].get_values()])

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
        self.y = y
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



# %%
if __name__ == '__main__':


    on_values = True
    stacked=True
    self = PlotData(data, stacked, on_values)





import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from pyAndy.auxiliary.aux_sql_func import write_sql

MONTH_DICT = {0:'JAN', 1:'FEB', 2:'MAR', 3:'APR', 4:'MAY', 5:'JUN',
               6:'JUL', 7:'AUG', 8:'SEP', 9:'OCT', 10:'NOV', 11:'DEC'}

SEASON_DICT = {'JAN': 'WINTER', 'FEB': 'WINTER', 'DEC': 'WINTER',
               'MAR': 'SPRING', 'APR': 'SPRING', 'MAY': 'SPRING',
               'JUN': 'SUMMER', 'JUL': 'SUMMER', 'AUG': 'SUMMER',
               'SEP': 'FALL', 'OCT': 'FALL', 'NOV': 'FALL'}
DOW_DICT = {0: 'MON', 1: 'TUE', 2: 'WED', 3: 'THU', 4: 'FRI', 5: 'SAT', 6: 'SUN'}
DOW_TYPE_DICT = {0: 'WEEKDAY', 1: 'WEEKDAY', 2: 'WEEKDAY', 3: 'WEEKDAY', 4: 'WEEKDAY', 5: 'SAT', 6: 'SUN'}

class TimeMap():

    def __init__(self, nhours=False, tm_filt=False, keep_datetime=False):

        self.tm_filt = tm_filt
        self.keep_datetime = keep_datetime

        self.df_time_red = pd.DataFrame()
        self.df_hoy_soy = pd.DataFrame()
        self.df_time_map = pd.DataFrame()

        if nhours:
            self.gen_soy_timemap(nhours)

    def gen_hoy_timemap(self, start='2015-1-1 00:00', stop='2015-12-31 23:59',
                        freq='H', tm_filt=False):

#        if __name__ == '__main__':
#            start='2015-1-1 00:00'
#            stop='2017-12-31 23:59'
#            freq='H'

        print('Generating time map from {} to {}'.format(start, stop))

        df_time_map = pd.DataFrame(index=pd.date_range(start, stop, freq=freq))
        df_time_map = df_time_map.reset_index().rename(columns={'index': 'DateTime'})

        df_time_map['month'] = df_time_map['DateTime'].dt.month
        df_time_map['mt_id'] = df_time_map['DateTime'].dt.month - 1
        df_time_map['mt'] = df_time_map['mt_id'].map(MONTH_DICT)
        df_time_map['season'] = df_time_map['mt'].map(SEASON_DICT)
        df_time_map['year'] = df_time_map['DateTime'].dt.year
        df_time_map['day'] = df_time_map['DateTime'].dt.day
        df_time_map['hour'] = df_time_map['DateTime'].dt.hour
        df_time_map['doy'] = df_time_map['DateTime'].dt.dayofyear
        df_time_map['wk_id'] = df_time_map['DateTime'].dt.week - 1
        df_time_map['wk'] = df_time_map['wk_id']
        df_time_map['dow'] = df_time_map['DateTime'].dt.weekday
        df_time_map['how'] = df_time_map['dow'] * 24 + df_time_map['hour']
        df_time_map['hom'] = (df_time_map['day'] - 1) * 24 + df_time_map['hour']
        df_time_map['dow_name'] = df_time_map['dow'].replace(DOW_DICT)
        df_time_map['dow_type'] = df_time_map['dow'].replace(DOW_TYPE_DICT)

        # week of the month
        dfwom = df_time_map[['wk_id', 'mt_id']]
        dfwkmt = df_time_map[['wk_id', 'mt_id']].pivot_table(index='wk_id', values=['mt_id'], aggfunc=np.median)
        dfwkmt = dfwkmt.rename(columns={'mt_id': 'wk_mt'})
        dfwom = dfwom.drop('mt_id', axis=1).drop_duplicates().join(dfwkmt, on=dfwkmt.index.names)
        dfwk_max = dfwom.pivot_table(index='wk_mt', values=['wk_id'], aggfunc=max).rename(columns={'wk_id': 'wk_max'}).reset_index() + 1
        dfwk_max = dfwk_max.set_index('wk_mt')
        dfwom = dfwom.join(dfwk_max, on=dfwk_max.index.names).fillna(0)
        dfwom['wom'] = dfwom['wk_id'] - dfwom['wk_max']
        dfwom = dfwom.set_index('wk_id')['wom']
        df_time_map = df_time_map.join(dfwom, on=dfwom.index.names)

        # add number of hours per week
        df_time_map = df_time_map \
                .join(df_time_map.pivot_table(values=['how'], index='wk',
                                              aggfunc=len).rename(columns={'how': 'wk_weight'}), on='wk')

        df_time_map_ndays = pd.DataFrame(df_time_map.loc[:,['mt', 'year', 'day']]
                                        .drop_duplicates()
                                        .pivot_table(values='day',
                                                     index=['year','mt'],
                                                           aggfunc=len))['day'].rename('ndays')
        df_time_map = df_time_map.join(df_time_map_ndays,
                                       on=df_time_map_ndays.index.names)

        # remove February 29
        mask_feb29 = ((df_time_map.mt == 'FEB') & (df_time_map.day == 29))
        df_time_map = df_time_map.loc[-mask_feb29].reset_index(drop=True)


        # add hour of the year column
        get_hoy = lambda x: x.reset_index(drop=True).reset_index().rename(columns={'index': 'hy'})[['hy']]
        df_time_map['hy'] = df_time_map.groupby(['year']).apply(get_hoy).reset_index(drop=True)['hy']

        # apply filtering
        mask = df_time_map.mt_id.apply(lambda x: True).rename('mask')
        if tm_filt:
            for ifilt in tm_filt:
                mask &= df_time_map[ifilt[0]].isin(ifilt[1])

        self.tm_filt_weight = mask.size / mask.sum()

        self.df_time_map = df_time_map.loc[mask].reset_index(drop=True)


        if not self.keep_datetime:
            self.df_time_map = self.df_time_map.drop('DateTime', axis=1)


    def gen_soy_timemap(self, nhours):

        if self.df_time_map.empty:
            self.gen_hoy_timemap(tm_filt=self.tm_filt)

        df_time_map = self.df_time_map

        # add soy column to dataframe, based on nhours
        len_rge = np.ceil(len(df_time_map)/nhours)
        len_rep = nhours
        df_tm = pd.DataFrame(np.repeat(np.arange(len_rge), [len_rep]),
                             columns=['sy']).iloc[:len(df_time_map)]
        df_time_map['sy'] = df_tm

        # add weight column to dataframe
        df_weight = df_time_map.pivot_table(values=['hy'], index='sy',
                                            aggfunc=len).rename(columns={'hy': 'weight'})
        df_time_map = df_time_map.join(df_weight, on='sy')

        self.df_hoy_soy = df_time_map[['sy', 'hy']]

        slct_col = [c for c in df_time_map.columns if not c in ['year', 'sy']]
        self.df_time_red = (df_time_map.reset_index()
                              .pivot_table(index=['year', 'sy'],
                                           values=slct_col,
                                           aggfunc=np.min).reset_index())


    def get_dst_days(self, list_months=['MAR', 'OCT']):

        # filter by relevant months
        _df = self.df_time_map.loc[self.df_time_map.mt.isin(list_months)]
        mask_dy = _df.dow_name == 'SUN'

        _df = _df.loc[mask_dy].pivot_table(index=['mt', 'year'], values='doy',
                              aggfunc=np.max).reset_index()[['year', 'mt', 'doy']]

        dict_dst = _df.set_index(['year', 'mt'])['doy'].to_dict()


        return dict_dst

if __name__ == '__main__':
    pass
    #    tm = TimeMap(1, tm_filt=[('wk_id', [27]), ('dow', [2])])
#
#    tm.gen_soy_timemap(4)
#
#    tm.df_time_red
#
#    tm.
#
#    write_sql(tm.df_time_red, 'storage1', 'public', 'temp_time_map_soy', 'replace')

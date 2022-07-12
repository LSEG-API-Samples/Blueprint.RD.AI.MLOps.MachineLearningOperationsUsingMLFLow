import mlflow
import eikon as ek
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os

from xgboost import plot_importance


def create_features(df, label=None):
    collist = df.columns.tolist()
    relcols = collist[:-2]
    X = df[relcols]
    if label:
        y = df[label].to_frame()
        return X, y
    return X


experiment_name = "XGBoost-mlflow"
experiment = mlflow.get_experiment_by_name(experiment_name)
if not experiment:
    experiment_id = mlflow.create_experiment(experiment_name)

if __name__ == '__main__':
    mlflow.set_experiment(experiment_name=experiment_name)
    mlflow.xgboost.autolog()
    if not os.path.exists('./data/rics_data.csv'):
        ek.set_app_key('your-app-key-here')

        allRics = ['IDMBKY=ECI',
                   'IDCARY=ECI',
                   'USCARS=ECI',
                   'ESCARM=ECI',
                   'ESCARY=ECI',
                   'ITCARM=ECI',
                   'aJPPNISS',
                   'aCATRKSBSP',
                   'aUSAUIMMXP',
                   'aGBACECARP',
                   'aLUNCAR',
                   'aISNVEHREG/A',
                   'aROACECARP']

        start = '2010-01-01'
        end = '2020-01-01'
        ts = pd.DataFrame()
        df = pd.DataFrame()

        for r in allRics:
            try:
                ts = ek.get_timeseries(r, start_date=start, end_date=end, interval='monthly')
                ts.rename(columns={'VALUE': r}, inplace=True)
                if len(ts):
                    df = pd.concat([df, ts], axis=1)
                else:
                    df = ts
            except:
                pass

        dfs = df.copy()
        lags = 6
        for r in dfs.columns.values:
            for lag in range(1, lags + 1):
                dfs[r + '_lag_{}'.format(lag)] = dfs[r].shift(lag)

        df1 = ek.get_timeseries(['BMWG.DE'], ['CLOSE'], start_date='2010-01-01', end_date='2020-01-01', interval='monthly')
        df2, err = ek.get_data('BMWG.DE', ['TR.RevenueMean(SDate=-3,EDate=-123,Period=FY1,Frq=CM).calcdate',
                                           'TR.RevenueMean(SDate=-3,EDate=-123,Period=FY1,Frq=M)'])
        df2['Calc Date'] = pd.to_datetime(df2['Calc Date'].str[:10])
        df2.set_index('Calc Date', inplace=True)
        df2.sort_index(ascending=True, inplace=True)

        dfs = pd.concat([dfs, df1], axis=1)
        dfs = pd.concat([dfs, df2['Revenue - Mean']], axis=1)

        b = list(dfs.columns[0:].values)
        for col in b:
            col_zscore = col + '_zscore'
            dfs[col_zscore] = (dfs[col] - dfs[col].mean()) / dfs[col].std(ddof=0)

        dfs = dfs.loc[:, dfs.columns.str.contains('zscore')]
        dfs = dfs.dropna()
        dfs.to_csv('./data/rics_data.csv')
    else:
        dfs = pd.read_csv('./data/rics_data.csv')

    split_date = '01-01-2015'
    pri_train = dfs.loc[dfs.index <= split_date].copy()
    pri_test = dfs.loc[dfs.index > split_date].copy()

    X_train, y_train = create_features(pri_train, label='CLOSE_zscore')
    X_test, y_test = create_features(pri_test, label='CLOSE_zscore')

    x_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    reg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=50, verbose=True)

    _ = plot_importance(reg, height=0.9)

    pri_test1 = pri_test.copy()
    pri_test1['ML_Prediction'] = reg.predict(X_test)
    pri_all = pd.concat([pri_test1, pri_train], sort=False)

    _ = pri_all[['CLOSE_zscore', 'ML_Prediction']].plot(figsize=(15, 5))
    plt.show()

    autolog_current = mlflow.last_active_run()

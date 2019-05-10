import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import util
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from scipy import stats
import pandas as pd
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=16)


def best_fit(X, Y):

    xbar = sum(X) / len(X)
    ybar = sum(Y) / len(Y)
    n = len(X)  # or len(Y)

    numer = sum([xi * yi for xi, yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    return a, b


def plot_hist(filename):
    df = util.csv_to_df(filename)
    class_labels = util.unique_classes(df)
    fig = plt.figure()
    fig, axes = plt.subplots(len(class_labels), 1, sharex='col', sharey='row')

    ax_count = 0
    for class_label in class_labels:
        observations = df.loc[class_label].values.flatten()
        sns.distplot(observations, ax=axes[ax_count], hist=True, kde=True,
                     label="1")
        ax_count += 1

    plt.show()


def plot_joint_hist(filename):
    df = util.csv_to_df(filename)
    class_labels = util.unique_classes(df)

    min_size = min(len(df.loc[1].values.flatten()),
                   len(df.loc[2].values.flatten()))
    sns.jointplot(df.loc[1].values.flatten()[:min_size], df.loc[
                  2].values.flatten()[:min_size], kind="kde")
    min_size = min(len(df.loc[1].values.flatten()),
                   len(df.loc[3].values.flatten()))
    sns.jointplot(df.loc[1].values.flatten()[:min_size], df.loc[
                  3].values.flatten()[:min_size], kind="kde")
    min_size = min(len(df.loc[2].values.flatten()),
                   len(df.loc[3].values.flatten()))
    sns.jointplot(df.loc[2].values.flatten()[:min_size], df.loc[
                  3].values.flatten()[:min_size], kind="kde")

    plt.show()


def plot_box(filename):
    df = util.csv_to_df(filename)
    class_labels = util.unique_classes(df)

    plt.boxplot([df.loc[1].values.flatten(),
                 df.loc[2].values.flatten(),
                 df.loc[3].values.flatten()])

    plt.show()


def plot_lag(filename):
    df = util.csv_to_df(filename)
    class_labels = util.unique_classes(df)
    fig = plt.figure()
    ax_count = 1
    for class_label in class_labels:
        # count change count to using class_label: assumes classes are ordered
        # 1-n though
        ax = fig.add_subplot(len(class_labels), 1, ax_count)
        class_mean, class_std = util.mean_std_of_class(df.loc[[class_label]])
        lag_plot(class_mean, lag=1, label=class_label)
        ax_count += 1
    plt.show()


def plot_ac(filename):
    df = util.csv_to_df(filename)
    class_labels = util.unique_classes(df)
    fig = plt.figure()
    class_mean, class_std = util.mean_std_of_class(df.loc[1])
    autocorrelation_plot(class_mean.append(class_mean), label='Cepheid')
    class_mean, class_std = util.mean_std_of_class(df.loc[2])
    autocorrelation_plot(class_mean.append(
        class_mean), label='Eclipsing Binary')
    class_mean, class_std = util.mean_std_of_class(df.loc[3])
    autocorrelation_plot(class_mean.append(class_mean), label='RR Lyrae')
    plt.show()

if __name__ == '__main__':

    # Cepheid Process
    # df = pd.read_csv('other/ce_dm.csv')
    # dcep_target = df[(df.type_best_classification ==
    #                   'DCEP') & (df.int_average_g < 16.34 + 0.3) & (df.int_average_g > 16.34 - 0.3)]
    # dcep = df[(df.type_best_classification ==
    #            'DCEP') & ((df.int_average_g >= 16.34 + 0.3) | (df.int_average_g <= 16.34 - 0.3))]

    # plt.figure(figsize=(8, 6))
    # plt.gca().invert_yaxis()

    # x = np.log(df[(df.type_best_classification ==
    #                'DCEP')][['p1']]).values.flatten()
    # y = df[(df.type_best_classification ==
    #         'DCEP')][['int_average_g']].values.flatten()
    # x_target = np.log(dcep_target[['p1']]).values.flatten()
    # y_target = dcep_target[['int_average_g']].values.flatten()

    # x_out = np.log(dcep[['p1']]).values.flatten()
    # y_out = dcep[['int_average_g']].values.flatten()

    # a, b = best_fit(x, y)
    # yfit = [a + b * xi for xi in x]
    # plt.scatter(x_out,y_out, color='palegreen',s=10, alpha=0.5, label='Known Cepheid Stars')

    # plt.scatter(x_target,y_target, color='g',s=100, alpha=0.5, label='Cepheid Stars with Similar Magnitude')
    # # plt.scatter(np.log(dcep_target['p1']), dcep_target[
    # #             'int_average_g'],s=100, color='g', alpha=0.5, label='Cepheid Stars with Similar Magnitude')
    # plt.plot(x, yfit, color='lime', alpha=0.5)

    # rect1 = plt.Rectangle((-2, 16.34 - 0.3), 6, 0.6, fill=True, alpha=0.2,
    #                      color='grey', linewidth=1, label='Luminosity Range')
    # rect2 = plt.Rectangle((0, 12), 1.15, 10, fill=True, alpha=0.2,
    #                       color='grey', linewidth=1, label='Possible Period')
    # rect3 = plt.Rectangle((2.75, 12), 0.08, 10, fill=True, alpha=0.2,
    #                       color='grey', linewidth=1)
    # plt.scatter(np.log(1.117), 16.34, color='gold',
    #            marker="*", s=300, label='Best Fit')
    # ax = plt.gca()
    # ax.add_patch(rect2)
    # plt.xlabel('log(Period) (log(day))')
    # plt.ylabel('Apparent Magnitude')
    # plt.title('Computed Result of our Star')
    # plt.legend()
    # plt.show()

    # RR Lyrae
    # df = pd.read_csv('other/rr_dm.csv')
    # rrc_target = df[(df.best_classification ==
    #                  'RRC') & (df.int_average_g < 19.110854 + 0.3) & (df.int_average_g > 19.110854 - 0.3)]
    # rrc = df[(df.best_classification ==
    #           'RRC') & ((df.int_average_g >= 19.110854 + 0.3) | (df.int_average_g <= 19.110854 - 0.3))]
    # rrab_target = df[(df.best_classification == 'RRAB') & (
    #     df.int_average_g < 19.110854 + 0.3) & (df.int_average_g > 19.110854 - 0.3)]
    # rrab = df[(df.best_classification == 'RRAB') & ((
    #     df.int_average_g >= 19.110854 + 0.3) | (df.int_average_g <= 19.110854 -
    #                                             0.3))]

    # plt.figure(figsize=(8, 5))
    # plt.gca().invert_yaxis()
    # plt.scatter(np.log(rrc_target['p1']), rrc_target[
    #             'int_average_g'], marker='*', color='r', alpha=0.4, label='RRC within range')
    # plt.scatter(np.log(rrc['p1']), rrc[
    #             'int_average_g'], color='r', alpha=0.2, label='RRC')
    # plt.scatter(np.log(rrab_target['p1']), rrab_target[
    #             'int_average_g'], marker='*', color='b', alpha=0.5, label='RRAB within range')
    # plt.scatter(np.log(rrab['p1']), rrab[
    #             'int_average_g'], color='b', alpha=0.2, label='RRAB')

    # rect = plt.Rectangle((-2, 19.110854 - 0.3), 6, 0.6, fill=True, alpha=0.2,
    #                      color='grey', linewidth=1, label='Data Average')
    # rect2 = plt.Rectangle((1, 12), 1.15, 10, fill=True, alpha=0.1,
    #                      color='grey', linewidth=1, label='Possible Period')
    # rect3 = plt.Rectangle((2.75, 12), 0.08, 10, fill=True, alpha=0.1,
    #                      color='grey', linewidth=1)
    # plt.scatter(np.log(0.629),19.110854,color='lime',marker="*",s=300,label='Best Fit')
    # ax = plt.gca()
    # ax.add_patch(rect2)
    # plt.xlabel('log(Period) (log(day)')
    # plt.ylabel('Apparent Magnitude')
    # plt.title('Computed Result of our RR Lyrae Star')
    # plt.legend()
    # plt.show()

    # Gaia Data
    # df = pd.read_csv('other/ce_dm.csv')
    # dcep = df[(df.type_best_classification ==
    #            'DCEP')]
    # acep = df[(df.type_best_classification ==
    #            'ACEP')]
    # t2cep = df[(df.type_best_classification ==
    #            'T2CEP')]

    # plt.figure(figsize=(8, 6))
    # plt.gca().invert_yaxis()

    # xd = np.log(df[(df.type_best_classification ==
    #                'DCEP')][['p1']]).values.flatten()
    # yd = df[(df.type_best_classification ==
    #         'DCEP')][['int_average_g']].values.flatten()
    # ad, bd = best_fit(xd, yd)
    # yfitd = [ad + bd * xi for xi in xd]

    # xa = np.log(df[(df.type_best_classification ==
    #                'ACEP')][['p1']]).values.flatten()
    # ya = df[(df.type_best_classification ==
    #         'ACEP')][['int_average_g']].values.flatten()
    # aa, ba = best_fit(xa, ya)

    # yfita = [aa + ba * xi for xi in xa]

    # xt = np.log(df[(df.type_best_classification ==
    #                'T2CEP')][['p1']]).values.flatten()
    # yt = df[(df.type_best_classification ==
    #         'T2CEP')][['int_average_g']].values.flatten()
    # at, bt = best_fit(xt, yt)
    # yfitt = [at + bt * xi for xi in xt]

    # plt.scatter(xd,yd, color='green',s=10, alpha=0.5, label='D-Cepheids')
    # plt.scatter(xa,ya, color='red',s=10, alpha=0.5, label='A-Cepheids')
    # plt.scatter(xt,yt, color='blue',s=10, alpha=0.5, label='T2-Cepheids')
    # # plt.scatter(np.log(dcep_target['p1']), dcep_target[
    # #             'int_average_g'],s=100, color='g', alpha=0.5, label='Cepheid Stars with Similar Magnitude')
    # plt.plot(xd, yfitd, color='green', alpha=0.5,label='D Fitting')
    # plt.plot(xa, yfita, color='red', alpha=0.5,label='A Fitting')
    # plt.plot(xt, yfitt, color='blue', alpha=0.5,label='T2 Fitting')

    # rect1 = plt.Rectangle((-2, 16.34 - 0.3), 6, 0.6, fill=True, alpha=0.2,
    #                      color='grey', linewidth=1)
    # rect2 = plt.Rectangle((0, 12), 1.15, 10, fill=True, alpha=0.2,
    #                       color='grey', linewidth=1, label='Possible Period')
    # rect3 = plt.Rectangle((2.75, 12), 0.08, 10, fill=True, alpha=0.2,
    #                       color='grey', linewidth=1)
    # #plt.scatter(np.log(1.117), 16.34, color='gold',
    #  #           marker="*", s=300, label='Best Fit')
    # ax = plt.gca()
    # ax.add_patch(rect1)
    # plt.xlabel('log(Period) (log(day))')
    # plt.ylabel('Apparent Magnitude')
    # plt.title('Cepheid Star Data in Gaia Archive')
    # plt.legend(loc='upper left')
    # plt.show()

    # # Residual
    # df = pd.read_csv('other/ce_dm.csv')
    # dcep_target = df[(df.type_best_classification ==
    #                   'DCEP') & (df.int_average_g < 16.34 + 0.3) & (df.int_average_g > 16.34 - 0.3)]
    # dcep = df[(df.type_best_classification ==
    #            'DCEP') & ((df.int_average_g >= 16.34 + 0.3) | (df.int_average_g <= 16.34 - 0.3))]

    # plt.figure(figsize=(8, 6))
    # plt.gca().invert_yaxis()

    # x = np.log(df[(df.type_best_classification ==
    #                'DCEP')][['p1']]).values.flatten()
    # y = df[(df.type_best_classification ==
    #         'DCEP')][['int_average_g']].values.flatten()
    # a, b = best_fit(x, y)
    # yfit = [a + b * xi for xi in x]
    # residual = []
    # for i in range(len(x)):
    #     residual.append(y[i]-yfit[i])
    # class_mean, class_std = util.mean_std_of_class(np.array(residual))
    # print(class_mean)
    # print(class_std)
    # interval = stats.norm.interval(0.95,class_mean,class_std)
    # print(interval)
    # plt.hlines(interval[0],-1,3,linestyles='dashed',label='95% Confidential Level')
    # plt.hlines(interval[1],-1,3,linestyles='dashed')
    # plt.scatter(x,residual, color='green',s=10, alpha=0.5, label='Residuals of Known Cepheid Stars')
    # # plt.scatter(np.log(dcep_target['p1']), dcep_target[
    # #             'int_average_g'],s=100, color='g', alpha=0.5, label='Cepheid Stars with Similar Magnitude')
    # #plt.plot(x, yfit, color='lime', alpha=0.5,label='Luminosity-Period Relation of Cepheid Stars')

    # rect1 = plt.Rectangle((-2, 16.34 - 0.3), 6, 0.6, fill=True, alpha=0.2,
    #                      color='grey', linewidth=1, label='Data Average')
    # rect2 = plt.Rectangle((0, 12), 1.15, 10, fill=True, alpha=0.2,
    #                       color='grey', linewidth=1, label='Possible Period')
    # rect3 = plt.Rectangle((2.75, 12), 0.08, 10, fill=True, alpha=0.2,
    #                       color='grey', linewidth=1)
    # plt.scatter(np.log(1.117), 16.34-a-b*np.log(1.117), color='gold', marker="*", s=300, label='Best Fit')
    # ax = plt.gca()
    # plt.xlabel('log(Period) (log(day))')
    # plt.ylabel('Residual')
    # plt.title('Residuals of Cepheid Star Data in Gaia Archive')
    # plt.legend()
    # plt.show()

    # # Comparison of Cepheid
    # data = pd.read_csv("gcvs/gcvs_ce.txt", sep="\t",
    #                    names=["luminosity", "period"])
    # data = data.dropna()
    # gaia = pd.read_csv('other/ce_dm.csv')

    # gaia_period = gaia['p1'].values.flatten()
    # gaia_luminosity = gaia['int_average_g'].values.flatten()

    # luminosity = data['luminosity'].values.flatten()
    # period = data['period'].values.flatten()

    # period = pd.Series(np.log(period), name="log(period)")
    # luminosity = pd.Series(luminosity, name="Apparent Magnitude")

    # gaia_period = pd.Series(np.log(gaia_period), name="log(period)")
    # gaia_luminosity = pd.Series(gaia_luminosity, name="Apparent Magnitude")

    # cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)
    # sns.jointplot(period, luminosity, xlim=(-1, 4), ylim=(
    #     20, 10), kind="kde", cmap=cmap, color="lightsalmon", shade_lowest=False)
    # cmap = sns.cubehelix_palette(start=5, light=1, as_cmap=True)

    # sns.jointplot(gaia_period, gaia_luminosity, xlim=(
    #     -1, 4), ylim=(20, 10), kind="kde", cmap=cmap, color="green", shade_lowest=False)
    # plt.text(-3, 19, "Gaia", size=16, color='green')
    # plt.text(-2, 18, "GCS", size=16, color='lightsalmon')
    # plt.show()

    # Our Result
    # plt.figure(figsize=(5,5))
    # plt.gca().invert_yaxis()
    # plt.scatter(np.log(1.117), 16.34, color='gold', marker="*", s=300, label='Our Data')
    # plt.xlim(-1, 4)
    # plt.ylim(20, 10)
    # plt.legend(['Our Data','sr Data','ur Data'])

    # plt.show()

    # Comparison of RR
    data = pd.read_csv("gcvs/gcvs_rr.txt", sep="\t",
                       names=["luminosity", "period"])
    data = data.dropna()
    gaia = pd.read_csv('other/rr_dm.csv')

    gaia_period = gaia['p1'].values.flatten()
    gaia_luminosity = gaia['int_average_g'].values.flatten()

    luminosity = data['luminosity'].values.flatten()
    period = data['period'].values.flatten()

    period = pd.Series(np.log(period), name="log(period)")
    luminosity = pd.Series(luminosity, name="Apparent Magnitude")

    gaia_period = pd.Series(np.log(gaia_period), name="log(period)")
    gaia_luminosity = pd.Series(gaia_luminosity, name="Apparent Magnitude")

    cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)
    sns.jointplot(period, luminosity, xlim=(-1.5, 0), ylim=(
        20, 10), kind="kde", cmap=cmap, color="lightsalmon", shade_lowest=False)
    cmap = sns.cubehelix_palette(start=5, light=1, as_cmap=True)

    sns.jointplot(gaia_period, gaia_luminosity, xlim=(
        -1.5, 0), ylim=(20, 10), kind="kde", cmap=cmap, color="green", shade_lowest=False)
    plt.text(-3, 19, "Gaia", size=16, color='green')
    plt.text(-5, 18, "GSC", size=16, color='lightsalmon')
    plt.show()

    # # Our Result
    # plt.figure(figsize=(5,5))
    # plt.gca().invert_yaxis()
    # plt.scatter(np.log(0.41), 19.11, color='gold', marker="*", s=300, label='Our Data')
    # plt.xlim(-1.5, 0)
    # plt.ylim(20, 10)
    # plt.legend(['Our Data','sr Data','ur Data'])

    # plt.show()

    # Comparison of Eclipsing Binaries
    # data = pd.read_csv("gcvs/gcvs_eb.txt", sep="\t",
    #                    names=["luminosity", "period"])
    # data = data.dropna()
    # gaia = pd.read_csv("gcvs/gaia_eb.txt", sep="\t",
    #                    names=["luminosity", "period"])
    # gaia = gaia.dropna()

    # gaia_period = gaia['period'].values.flatten()
    # gaia_luminosity = gaia['luminosity'].values.flatten()

    # luminosity = data['luminosity'].values.flatten()
    # period = data['period'].values.flatten()
    # print(period)

    # period = pd.Series(np.log(period), name="log(period)")
    # luminosity = pd.Series(luminosity, name="Apparent Magnitude")

    # gaia_period = pd.Series(np.log(gaia_period), name="log(period)")
    # gaia_luminosity = pd.Series(gaia_luminosity, name="Apparent Magnitude")

    # cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)
    # sns.jointplot(period, luminosity,  kind="kde", cmap=cmap,
    #               color="lightsalmon", xlim=(-5, 4), ylim=(20, 10), shade_lowest=False)
    # cmap = sns.cubehelix_palette(start=5, light=1, as_cmap=True)

    # sns.jointplot(gaia_period, gaia_luminosity,  kind="kde",
    #               cmap=cmap, color="green", xlim=(-5, 4), ylim=(20, 10), shade_lowest=False)
    # plt.text(-8, 19, "Gaia", size=16, color='green')
    # plt.show()
    
    # Our Result
    # plt.figure(figsize=(5,5))
    # plt.gca().invert_yaxis()
    # plt.scatter(np.log(1.117), 17.77, color='gold', marker="*", s=300, label='Our Data')
    # plt.xlim(-5, 4)
    # plt.ylim(20, 10)
    # plt.legend(['Our Data','sr Data','ur Data'])

    # plt.show()
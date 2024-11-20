import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

def boxplot_(data,col):
    sn.boxplot(data[col])
    plt.title('{}'.format(col))
    return plt.show()

def bar_plot(data,column_name):
    data[column_name].value_counts(normalize=True).plot(kind="bar", color="orange")
    plt.ylabel('proportion')
    plt.title("Distribution of {}" .format(column_name))
    return plt.show()


def boxplot_biv(data,col):
    sn.boxplot(x=target_name, y=col,data=data)
    plt.xlabel(target_name)
    plt.ylabel('{}'.format(col))
    return plt.show()


def bar_plot_biv(data,col,target_name="stroke"):
    for value in np.unique(data[target_name]):
        data.loc[data[target_name]==value,col].value_counts(normalize=True).plot(kind='bar', color="green")
        plt.title("bar of {} when stroke is {} ".format(col, value) )
        plt.show()

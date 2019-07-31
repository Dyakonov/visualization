# библиотека для визуализации опросов
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from scipy.stats import gaussian_kde

def plot_density(x, color='white', label='плотность', covariance_factor=0.15):
    """
    нарисовать плотность признака x
    """
    density = gaussian_kde(x)
    xs = np.linspace(np.min(x), np.max(x), 100)
    density.covariance_factor = lambda : covariance_factor
    density._compute_covariance()
    plt.plot(xs, density(xs), lw=2, c=color, label=label)
    
    
def show_bar(d, show_percent=True, rotation=0, eps_x = 0.0, eps_y = 0.0, alpha=1.0, edgecolor='white', color='#000099', new_figure=True):
    """
    показать столбцовую диаграмму ответов
    """
    if new_figure:
        plt.figure(figsize=(7, 3))
    plt.bar(np.arange(len(d)), d.values, color=color, edgecolor=edgecolor, alpha=alpha)
    plt.ylabel('число ответивших')
    plt.xticks(np.arange(len(d)), d.index, rotation=rotation);
    plt.grid(lw=2, axis='y')
    if show_percent:
        d_normed = np.round(d / d.values.sum() * 100, 1)
        for i in range(len(d)):
            plt.text(i + eps_x, d[i] + eps_y, '{}%'.format(d_normed.values[i]), color=color, weight='bold', fontsize=14)

def show_barh(d, show_percent=True, figsize=(7, 3), eps_x = 0.0, eps_y = 0.0, alpha=1.0, edgecolor='white', reverse=False):
    """
    нарисовать горизонтальную
    столбцовую диаграмму
    """
    plt.figure(figsize=figsize)
    indexes = np.arange(len(d))
    if reverse:
        indexes = -indexes
    plt.barh(indexes, d.values, color='#000099', edgecolor=edgecolor, alpha=alpha)
    plt.yticks(indexes, d.index);
    plt.grid(lw=2, axis='x')
    if show_percent:
        d_normed = np.round(d / d.values.sum() * 100, 1)
        for i in range(len(d)):
            plt.text(d[i] + eps_x, indexes[i] + eps_y, '{}%'.format(d_normed.values[i]), color='#000099', weight='bold', fontsize=14)
    
def make_sums(data, dct):
    """
    проссумировать ответы,
    если вопрос разбит на несколько
    """
    l = []
    n = []
    for name in dct:
        l.append((data[name]==1.).sum())
        n.append(dct[name])
        print ('{} / {}'.format((data[name]==1.).sum(), (data[name]==0.).sum()), dct[name])
    d = pd.Series(l, index = n)
    d.sort_values(inplace=True)
    return (d)   

def show_tab(data, name, dct=None, show_percent=True):
    """
    показать статистику по одному признаку
    """
    if dct is None:
        s = data[name]
    else:
        s = data[name].map(dct)
    if show_percent:
        df = pd.concat([s.value_counts(), (100 * s.value_counts(normalize=True)).round(1).astype(str) + '%'], axis=1)
        df.columns = ['ответов', 'процентов']
    else:
        df = pd.DataFrame(s.value_counts())
    return df

def cross_plus_percent(d):
    """
    Добавить проценты в crosstab
    """
    return d.astype(str) + ' (' + (100 *d / np.sum(d.values)).round(1).astype(str) + '%)'

def show_probabilities(data, name, target, target_0=2, target_1=1):
    """
    вероятности целевого
    по категориям выбранного признака
    """
    tmp = data[data[target].isin([target_0, target_1])][[name, target]]
    tmp[target] = tmp[target].map({target_0: 0, target_1: 1})
    return pd.DataFrame(tmp.groupby(name)[target].mean())

def show_confusion_matrix(cm):
    """
    показать матрицу несоответствий
    """
    # df = pd.DataFrame(confusion_matrix([0,0,0,1,1,1,1], [0,1,1,1,1,1,1]))
    df = pd.DataFrame(cm)
    df = pd.DataFrame([['TN = ', 'FP = '],['FN = ', 'TP = ']]).astype(str) + df.astype(str)
    df.index = [r'$y=0$', r'$y=1$']
    df.columns = [r'$a=0$', r'$a=1$']
    return (df)
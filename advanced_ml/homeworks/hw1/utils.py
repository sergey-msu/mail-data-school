import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import warnings
warnings.filterwarnings('ignore')


TRAIN_DAYS = 50
MAY_1_DAY = 59
JUNE_1_DAY = 90
SEPT_1_DAY = 182


def prepare_x(x, dim):
    X = np.ones((x.shape[0], dim))
    for i in range(1, dim):
        X[:, i] = x**i
    return X


def plot_distribution(mean, cov, lims, title, axis, N=500, xlabel='w0', ylabel='w1'):
    """ Строит распределение параметров. """
    X = np.linspace(*lims[0], N)
    Y = np.linspace(*lims[1], N)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    Z = multivariate_normal.pdf(pos, mean=mean, cov=cov)
    axis.pcolormesh(X, Y, Z, cmap=plt.cm.jet)
    axis.set_xlim(lims[0])
    axis.set_ylim(lims[1])
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    
    
def sample_exponents(data, model, country, figsize=(13.5, 6)):
    """ Семплирует экспоненты из распределения параметров модели. """
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    gs = axes[0, 0].get_gridspec()
    for ax in axes[:, 0]:
        ax.remove()
    axbig = fig.add_subplot(gs[:, 0])

    may_1_samples = []
    june_1_samples = []
    sept_1_samples = []
    xs = data['day'].values
    ws = np.random.multivariate_normal(model.mu_N, model.sigma_N, 3000)

    for w in ws:
        ys = np.exp(w[0] + w[1]*xs)
        may_1_samples.append(ys[MAY_1_DAY])
        june_1_samples.append(ys[JUNE_1_DAY])
        sept_1_samples.append(ys[SEPT_1_DAY])
        axbig.plot(xs, ys, 'k-', lw=.4, color='red', alpha=0.01)
        axbig.set_ylim((0, 1e6))

    preds = [np.exp(model.predict(np.array([d]))) for d in data['day']]
    
    axbig.axvline(x=TRAIN_DAYS, color='orange', alpha=0.9, label='train upper')
    axbig.plot(data['total_cases'].values)
    axbig.set_title(f'Всего случаев: {country}')
    axbig.set_xlabel('дни')
    axbig.set_ylabel('число')
    axbig.legend()

    axes[0][1].hist(may_1_samples, color='red', bins=50, label='семпл')
    axes[0][1].axvline(x=data['total_cases'].values[MAY_1_DAY], label='реальность')
    axes[0][1].axvline(x=preds[MAY_1_DAY], ls='--', color='green', label='предсказание')
    axes[0][1].set_title('1 Мая')
    axes[0][1].legend()

    axes[1][1].hist(june_1_samples, color='red', bins=50, label='семпл')
    axes[1][1].axvline(x=data['total_cases'].values[JUNE_1_DAY], label='реальность')
    axes[1][1].axvline(x=preds[JUNE_1_DAY], ls='--', color='green', label='предсказание')
    axes[1][1].set_title('1 Июня')
    axes[1][1].legend()

    axes[2][1].hist(sept_1_samples, color='red', bins=50, label='семпл')
    axes[2][1].axvline(x=data['total_cases'].values[SEPT_1_DAY], label='реальность')
    axes[2][1].axvline(x=preds[SEPT_1_DAY], ls='--', color='green', label='предсказание')
    axes[2][1].set_title('1 Сентября')
    axes[2][1].legend()

    plt.tight_layout()
    plt.show()
    
    result = {
        'preds': preds,
        'may_1': may_1_samples,
        'june_1': june_1_samples,
        'sept_1': sept_1_samples,
        
    }
    
    return result


def sample_errfs(data, model, country, days, day_labels, n_samples=3000, figsize=(13.5, 6)):
    """ Семплирует функции ошибок из распределения параметров модели. """
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    gs = axes[0, 0].get_gridspec()
    for ax in axes[:, 0]:
        ax.remove()
    axbig = fig.add_subplot(gs[:, 0])

    xs = data['day'].values
    ws = np.random.multivariate_normal(model.mu_N, model.sigma_N, n_samples)
    samples = np.zeros((len(data), n_samples))

    for i, w in enumerate(ws):
        ys = np.cumsum(np.exp(w[0] + w[1]*xs + w[2]*(xs**2)))
        samples[:, i] = ys
        axbig.plot(xs, ys, 'k-', lw=.4, color='red', alpha=0.1)
        axbig.set_ylim((0, 1e6))
    
    preds = np.cumsum([np.exp(model.predict(np.array([d]))) for d in data['day']])

    for i in range(3):
        day = days[i]
        day_label = day_labels[i]
        day_samples = samples[day, :]
        day_samples_95 = np.percentile(day_samples, 95)
        day_samples = day_samples[day_samples < day_samples_95]
        axes[i][1].hist(day_samples, color='red', bins=50, label='семпл')
        axes[i][1].axvline(x=data['total_cases'].values[day], label='реальность')
        axes[i][1].axvline(x=preds[day], ls='--', color='green', label='предсказание')
        axes[i][1].set_title(day_label)
        axes[i][1].legend()

    axbig.axvline(x=TRAIN_DAYS, color='orange', alpha=0.9, label='train upper')
    axbig.plot(data['total_cases'].values, lw=2, label='реальность')
    axbig.plot(preds, '--', lw=2, color='green', label='предсказание')
    axbig.fill_between(data['day'].values,
                       np.percentile(samples, 10, axis=1), 
                       np.percentile(samples, 90, axis=1), 
                       color='green', label='10-90 перцентиль', alpha=0.3)
    axbig.set_title(f'Всего случаев: {country}')
    axbig.set_xlabel('дни')
    axbig.set_ylabel('число')
    axbig.legend()

    plt.tight_layout()
    plt.show()

    return preds, samples


def plot_parameter_cloud(params):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    names = params.columns[:-1]
    
    axes[0].scatter(params.iloc[:,0], params.iloc[:,1], alpha=0.5)
    axes[0].set_xlabel(names[0])
    axes[0].set_ylabel(names[1])
    
    axes[1].scatter(params.iloc[:,0], params.iloc[:,2], alpha=0.5)
    axes[1].set_xlabel(names[0])
    axes[1].set_ylabel(names[2])
    
    axes[2].scatter(params.iloc[:,1], params.iloc[:,2], alpha=0.5)
    axes[2].set_xlabel(names[1])
    axes[2].set_ylabel(names[2])
    
    plt.tight_layout()
    plt.show()
import pandas as pd
import random
import numpy as np
from tqdm import tqdm_notebook as tqdm 
import warnings 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import json
warnings.filterwarnings('ignore')

import matplotlib as mpl
mpl.rcParams['figure.dpi'] =100

import adaptive
adaptive.notebook_extension()

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from diversipy import *
from scipy.stats import skewnorm
from matplotlib.ticker import MaxNLocator


def split(X, y, sample, trainval_size, icvconfs, total_sims, sim_steps, noise_level=0, noise_type = 'Gaussian'): 

    trainval_samples = [icvconfs[tuple(x)] for x in sample]
    trainval_idx = []
    for counter, idx in enumerate(trainval_samples): 
        trainval_idx.extend(list(range(idx*sim_steps,(idx+1)*sim_steps)))
        
    test_idx = list(set(range(total_sims*sim_steps)) - set(trainval_idx))
    
    X_test, y_test = X[test_idx], y[test_idx]
    X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]

    scaler_X = MinMaxScaler().fit(X_trainval)
    X_trainval = scaler_X.transform(X_trainval)
    X_test = scaler_X.transform(X_test)
    scaler_y = MinMaxScaler().fit(y_trainval)
    y_trainval = scaler_y.transform(y_trainval)
    y_test = scaler_y.transform(y_test)
    
    noise = None
    
    if noise_type == 'Gaussian':
        print('Gaussian Noise is Added ...')
        mu, sigma = 0, noise_level*np.std(y_trainval, axis=0)
        noise = np.random.normal(mu, sigma, y_trainval.shape)
        y_trainval = y_trainval + noise
    elif noise_type == 'Uniform':
        print('Uniform Noise is Added ...')
        mu, sigma = 0, noise_level*np.std(y_trainval, axis=0)
        noise = np.random.uniform(-sigma, sigma, y_trainval.shape)
        y_trainval = y_trainval + noise
        
    return X_trainval, X_test, scaler_X, y_trainval, y_test, scaler_y


def random_sampler(trainval_size, total_sims, allsamples):
    return np.array([allsamples[x] for x in np.random.choice(total_sims, size=trainval_size, replace=False)])

def lhs_sampler(trainval_size, xmin, xmax, dim=3):
    
    doe = hycusampling.improved_lhd_matrix(trainval_size, dimension=dim, num_candidates=10000)/(trainval_size-1)
    
    sample = xmin + (xmax-xmin) * doe
    sample = np.round(sample).astype(int)
    sample = sample[~np.all(sample == 0, axis=1)]
    
    return sample

def adaptive_sampler(trainval_size, xmin, xmax, sim_steps, allsamples, icvconfs, groundtruth, output_size=4, input_size=3):
    
    trainval_size0 = trainval_size
    learner = adaptive.LearnerND(groundtruth, bounds=input_size*[(xmin, xmax)])
    learner.tell(tuple(input_size*[0]), output_size*[0])
    losses = []
    
    for i in range(trainval_size):
        try:
            point, _ = learner._ask()
            point = tuple(np.round(j) for j in point)
            
            if list(point) in learner.points.tolist():
                remaining_points = [k for k in allsamples if k not in learner.points.tolist()]
                r = np.random.randint(len(remaining_points))
                point = tuple(remaining_points[r])
                
            value = groundtruth(point, icvconfs, sim_steps)    
            learner.tell(point, value)
            
        except: 
            print('Repeated Point ...')
            remaining_points = [k for k in allsamples if k not in learner.points.tolist()]
            r = np.random.randint(len(remaining_points))
            point = tuple(remaining_points[r])
            value = groundtruth(point, icvconfs, sim_steps)    
            learner.tell(point, value)
            
        losses.append(learner.loss())
    
    grads = np.gradient(losses)
    sample = learner.points.astype(int)
    sample = sample[~np.all(sample == 0, axis=1)]
    
    assert len(np.unique(sample, axis=0)) == trainval_size0, 'Adaptive Sampling Incurred Redundant Points ... Review Your Setup'
    
    return sample, losses, grads

def hybrid_sampler(trainval_size, xmin, xmax, sim_steps, allsamples, icvconfs, groundtruth, output_size=4, input_size=3, init_points = [[1,1,0], [0,1,1], [1,0,1], [1,1,1]]):
    
    trainval_size0 = trainval_size
    learner = adaptive.LearnerND(groundtruth, bounds=input_size*[(xmin, xmax)])
    learner.tell(tuple(input_size*[0]), output_size*[0])
    losses = []
    
    for point in init_points: 
        point = tuple([p*xmax for p in point])
        value = groundtruth(point, icvconfs, sim_steps)
        learner.tell(point, value)
        losses.append(learner.loss())
    trainval_size -= len(init_points)

    if trainval_size > 0:
        learner._bounds_points = []
        for i in range(trainval_size):
            try:
                point, _ = learner._ask()
                point = tuple(np.round(j) for j in point)

                if list(point) in learner.points.tolist():
                    remaining_points = [k for k in allsamples if k not in learner.points.tolist()]
                    r = np.random.randint(len(remaining_points))
                    point = tuple(remaining_points[r])

                value = groundtruth(point, icvconfs, sim_steps)    
                learner.tell(point, value)

            except: 
                print('Repeated Point ...')
                remaining_points = [k for k in allsamples if k not in learner.points.tolist()]
                r = np.random.randint(len(remaining_points))
                point = tuple(remaining_points[r])
                value = groundtruth(point, icvconfs, sim_steps)    
                learner.tell(point, value)
            
            losses.append(learner.loss())
    
    grads = np.gradient(losses)
    sample = learner.points.astype(int)
    sample = sample[~np.all(sample == 0, axis=1)]
    
    assert len(np.unique(sample, axis=0)) == trainval_size0, 'Adaptive Sampling Incurred Redundant Points ... Review Your Setup'
    
    return sample, losses, grads

def plot_samples(sample_list, color_list=['darkorange'], label_list=[''], fig_name='figure'):
    for sample, color, label in zip(sample_list, color_list, label_list):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('ICV #1')
        ax.set_ylabel('ICV #2')
        ax.set_zlabel('ICV #3')
    
        sample_new = np.vstack((sample, [0,0,0]))
        ax.scatter(sample_new[:,0], sample_new[:,1], sample_new[:,2], color=color, label=label)
        plt.legend()
        plt.show()
        fig.savefig(f'figs/{label}_{len(sample)}.png', dpi=200)

def plot_output(X_test, y_test, y_pred, scaler_X, scaler_y, total_sims, sim_steps, sample):
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=[10,5]) 
    X_ti = scaler_X.inverse_transform(X_test)
    y_ti = scaler_y.inverse_transform(y_test)
    y_pi = scaler_y.inverse_transform(y_pred)
    for row in ax: 
        for col in row:
            ts = np.random.randint(total_sims-len(sample), size=1)[0]
            col.plot(X_test[0:sim_steps,0], y_test[sim_steps*ts:sim_steps*(ts+1)], '--g', label='True')
            col.plot(X_test[0:sim_steps,0], y_pred[sim_steps*ts:sim_steps*(ts+1)], 'darkorange', label='Prediction')
    plt.show()

def plot_losses(losses, grads, filename='jsons/caseA/losses_x.png'):

    fig ,ax1 = plt.subplots()
    color = 'green'
    ax1.plot(range(len(losses)), losses, color=color)
    ax1.set_ylabel('Average Simplex Hypervolume', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlabel('Training Simulation Runs')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    color = 'red'
    ax2 = ax1.twinx()
    ax2.plot(range(len(grads)), grads, color=color)
    ax2.set_ylabel('Simplex Hypervolume Gradient', color = color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xlabel('Year')
    fig.savefig(filename, dpi=300, bbox_inches='tight')

def train_test_mlp(X_trainval, y_trainval, X_test, y_test):
    mlp = MLPRegressor((100, 100, 100, 100), validation_fraction=0)
    params = {'alpha' : np.logspace(-5,0,5)}
    mlp = GridSearchCV(mlp, params, cv=3, n_jobs=4)
    mlp.fit(X_trainval, y_trainval)
    y_pred = mlp.predict(X_test)
    score = r2_score(y_test, y_pred)
    return mlp, score, y_pred

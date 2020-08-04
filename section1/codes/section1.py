from sklearn.datasets import load_iris
import pandas as pd
from sklearn import mixture
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from codes.RLmaze.env import MazeEnv
from codes.RLmaze.agents import Agent
from codes.RLmaze.policies import EpsilonGreedy, Action_value
from codes.RLmaze.optimizers import Qlearning_optimizer
from codes.RLmaze.planning import ReplayModel_plus, step

colors = ['red', 'blue', 'green']

##### 1-1 
def get_and_show_iris_data(iris):
    iris_data = pd.DataFrame(iris.data[::25, 1:3], columns=["がく片の厚さ(cm)","花びらの長さ(cm)"])
    iris_target = pd.DataFrame(iris.target[::25], columns=["花の種類(0:setosa種, 1:versicolor種, 2:virginica種)"])
    iris_all = pd.concat([iris_data,iris_target], axis=1)
    return iris_all.head(6)

def get_data_with_supervision(iris, offset=0):
    ''' iris.data is array with shape=(150, 4).
        first 50 samples are in class 0,
        second 50 samples are in class 1,
        third 50 samples are in class 2.
    '''
    return iris.data[offset::2,1:3], iris.target[offset::2]

def get_data_wo_supervision(iris, offset=0):
    return iris.data[offset::2,1:3]

def plot_iris(x, y, offset=0, ax=None, marker='o'):
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot()
    ax.set_xlabel("sepal width (cm)", fontsize=15)
    ax.set_ylabel("petal length (cm)", fontsize=15)
    ax.set_title(r"$\{{\bf x}_n\}_{n=%d,%d,...,%d}$"%(offset, offset+2, offset+148))
    if y is not None:
        for label_index in range(3):
            ax.scatter(x[y==label_index,0], x[y==label_index,1], color=colors[label_index], marker=marker, label="y=%d"%label_index)
        ax.legend()
    else:
        ax.scatter(x[:,0], x[:,1], marker=marker, color='black')
    
    

def plot_density(gmm_model, ax, color='Purple', xrange=[2, 4.5], yrange=[1,7]):
    x = np.linspace(xrange[0], xrange[1])
    y = np.linspace(yrange[0], yrange[1])
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -gmm_model.score_samples(XX)+1
    Z = Z.reshape(X.shape)
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot()
    ax.set_xlabel("sepal width (cm)", fontsize=15)
    ax.set_ylabel("petal length (cm)", fontsize=15)
    ax.set_title("density contour")
    ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(yrange[0], yrange[1])
    ax.contour(X, Y, Z, norm=LogNorm(vmin=1, vmax=50.0), levels=np.logspace(0, 1, 5), colors=[color],
              alpha=0.8, label='wow')
    
    
def get_trained_GMM_model(data, n_components=1):
    model = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
    model.fit(data)
    return model

def return_densitylist_trained_by(x, y):
    fig = plt.figure(figsize=(15,5))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    plot_iris(x, y, ax=ax1)
    models=[]
    for label in range(3):
        conditioned_x = x[25*label:25*(label+1)]
        model=get_trained_GMM_model(conditioned_x)
        plot_density(model, ax=ax2, color=colors[label])
        models.append(model)
    return models

def erase(ax):
    ax.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
    ax.tick_params(bottom=False,
                left=False,
                right=False,
                top=False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

            


def plot_data_and_densitylist(x, y, models):
    fig = plt.figure(figsize=(15,5))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    plot_iris(x, y, ax=ax1, marker='x')
    for label in range(3):
        plot_density(models[label], ax=ax1, color=colors[label])
    ### erasing ax2 for rendering
    erase(ax2)


def return_density_trained_by(x):
    fig = plt.figure(figsize=(15,5))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    plot_iris(x, y=None, ax=ax1)
    for label in range(1):
        model=get_trained_GMM_model(x, n_components=3)
        plot_density(model, ax=ax2, color='black')
    return model

def plot_data_and_density(x, model):
    fig = plt.figure(figsize=(15,5))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    plot_iris(x, y=None, ax=ax1, marker='x')
    for label in range(1):
        plot_density(model, ax=ax1, color='black')
    ### erasing ax2 for rendering
    erase(ax2)

##### RL
def run_RL(size=10, T=1000):
    env = MazeEnv(size, size, threshold=1.3)
    q = Action_value(env, init=0); Agt = Agent(EpsilonGreedy(env, Q=q, epsilon=0.1))
    Opt = Qlearning_optimizer(Agt, eta=1., gamma=.95)
    Model = ReplayModel_plus(env)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1,2,1)
    ay = fig.add_subplot(1,2,2)

    env.render(fig=fig, ax=ax, canvas=True)
    env.reset()
    for t in range(T):
        step(env, q, Agt, Opt, Model)

    env.state = None
    env.render(fig=fig, ax=ay, values_table=q.values_table)

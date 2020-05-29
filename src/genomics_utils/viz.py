import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'make_learning_curve'
]


def make_learning_curve(dataset_name, model_name, losses, dpi=100):
    f = plt.figure(figsize=(9, 6), dpi=dpi)
    xs = np.arange(losses.shape[0])
    mean = np.mean(losses, axis=1)
    lower = np.quantile(losses, axis=1, q=0.1)
    upper = np.quantile(losses, axis=1, q=0.9)
    
    mean_line, = plt.plot(
        xs, mean,
        label='%s: mean loss' % (model_name, ),
        rasterized=True
    )
    plt.fill_between(
        xs, lower, upper,
        label='%s: 10%%-90%% percentiles' % (model_name, ),
        rasterized=True,
        color=mean_line.get_color(),
        alpha=0.25
    )
    
    
    plt.title(dataset_name)
    plt.xlabel('epoch')
    plt.ylabel('cross-entropy')
    plt.legend()
    
    return f

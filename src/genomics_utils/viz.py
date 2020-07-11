import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

__all__ = [
    'make_learning_curve',
    'make_coalescent_heatmap'
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
    plt.ylabel('KL loss')
    plt.legend()
    
    return f


def make_coalescent_heatmap(model_name, averaged_data_tuple, dpi=300):
    # f = plt.figure(, dpi=dpi)
    f, ax = plt.subplots(1, 1, figsize=(200, 10), dpi=dpi)
    im0 = ax.imshow(averaged_data_tuple[0], cmap='Wistia')
    ax.plot(averaged_data_tuple[1], lw=1, c='black')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax)

    plt.suptitle("Coalescent heatmap distribution by {} model".format(model_name), fontsize=20)
    ax.set_title("Softmax")
    plt.xlabel('site position')
    plt.ylabel('')
    ax.yaxis.set_ticks(np.arange(0, 20, step=1))
    return f

import pickle
from pathlib import Path

import matplotlib.pyplot as plt


class History():
    def __init__(self, history = None):
        self.history = None if history is None else history.history
    
    def save_history(self, path:str = "history.pickle"):
        with open(str(Path(path)), 'wb') as file:
            pickle.dump(self.history, file)
        return self

    def load_history(self, path:str = "history.pickle"):
        with open(str(Path(path)), 'rb') as file:
            self.history = pickle.load(file)
        return self
    
    def save_all(self, path:str = "plots"):
        for key in self.history:
            self.save_plot_history(path, [key])
        return self

    
    def show_history(self, metrics = ["loss", "precision", "recall", "auc", "accuracy"]):
        for metric in metrics:
            try:
                plt.plot(self.history[metric])
                plt.title("Model " + metric)
                plt.ylabel(metric)
                plt.xlabel("Epoch")
                plt.show()
                plt.clf()
            except Exception as e:
                print("Cannot create [" + metric + "] value plot")
                print(e)
        return self

    def show_all_history(self, metrics = ["loss", "val_loss"], limit:list = (0.0, 1.0)):
        plt.title("Model " + str(metrics))
        plt.xlabel("Epoch")
        plt.ylabel("Value of")
        plt.ylim(limit)
        for metric in metrics:
            try:
                plt.plot(self.history[metric], label=metric)
            except Exception as e:
                print("Cannot create [" + metric + "] value plot")
                print(e)
        plt.legend(bbox_to_anchor=(0., 1.07, 1., .107), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
        plt.show()
        plt.clf()
        return self

    def save_plot_history(self, path:str = "plots", metrics = ["loss", "precision", "recall", "auc", "accuracy"]):
        for metric in metrics:
            try:
                plt.plot(self.history[metric])
                plt.title("Model " + metric)
                plt.ylabel(metric)
                plt.xlabel("Epoch")
                plt.savefig(str(Path(path) / (metric + '.png')))
                plt.clf()
            except Exception as e:
                print("Cannot create [" + metric + "] value plot")
        return self

if __name__ == '__main__':
    hist = History()
    hist.load_history('history/20201120_0011_model_unet_2d_1.pickle')
    hist.show_all_history(['loss', 'val_loss'], (0.0, 0.02))
    hist.save_all('history/plots/')
    #hist.show_all_history(['val_accuracy', 'accuracy'])
    #hist.save_plot_history('history/plots/')

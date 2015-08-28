from matplotlib import lines
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from ui_oracle.transform_oracle import TransformOracle

__author__ = 'maeglin89273'


class LearningOracle:
    def __init__(self):
        self.wavelet = "coif4"

    def train(self, signals, targets):
        self.transform_signals(signals)

        signals, targets = self.to_arrays(signals, targets)

        signals_train, signals_test, targets_train, targets_test = train_test_split(signals, targets, random_state=42)

        # ml_algorithm = KNeighborsClassifier()
        self.ml_algorithm = SVC(kernel="poly")
        self.ml_algorithm.fit(signals_train, targets_train)
        print(self.ml_algorithm.score(signals_train, targets_train))
        print(self.ml_algorithm.score(signals_test, targets_test))
        return self.ml_algorithm.score(signals, targets)



    def pca_plot3d(self, signals, targets):
        self.transform_signals(signals)
        signals, targets = self.to_arrays(signals, targets)
        rand_pca = RandomizedPCA(n_components=3)
        rand_pca.fit(signals)
        plot_values = rand_pca.transform(signals)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        collection = ax.scatter(plot_values[:, 0], plot_values[:, 1], plot_values[:, 2], c=targets)
        self.addLegend(ax, collection)
        plt.show()
        plt.close()

    def pca_plot2d(self, signals, targets):
        self.transform_signals(signals)
        signals, targets = self.to_arrays(signals, targets)
        rand_pca = RandomizedPCA(n_components=2)
        rand_pca.fit(signals)
        plot_values = rand_pca.transform(signals)

        collection = plt.scatter(plot_values[:, 0], plot_values[:, 1], c=targets)

        self.addLegend(plt, collection)
        plt.show()
        plt.close()

    def addLegend(self, axis, collection):
        scatter_proxies = []
        labels = []
        for tag in self.label_encoder.classes_:
            scatter_proxies.append(lines.Line2D([0], [0], linestyle="none", marker="o",
                                                c=collection.to_rgba(self.label_encoder.transform(tag))))
            labels.append(tag)
        axis.legend(scatter_proxies, labels, numpoints=1)


    def transform_signals(self, signals):
        # todo: make transformation selective
        signal_tags = signals[0].keys()
        for i, signal in enumerate(signals):
            transformed_signal = []
            for tag in signal_tags:
                transformed_single_signal = np.absolute(np.fft.rfft(signal[tag]))
                transformed_signal.append(
                    transformed_single_signal[TransformOracle.FFT_START_FREQUENCY:len(transformed_single_signal) // 2])
            signals[i] = transformed_signal
        return signals

    def transform_signal(self, signal):
        transformed_signal = []
        for tag in signal:
            transformed_single_signal = np.absolute(np.fft.rfft(signal[tag]))
            transformed_signal.append(
                transformed_single_signal[TransformOracle.FFT_START_FREQUENCY:len(transformed_single_signal) // 2])
        return np.hstack(transformed_signal)


    def to_arrays(self, signals, targets):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(targets)
        targets = self.label_encoder.transform(targets)
        signals = np.array(signals)
        return signals.reshape(signals.shape[0], -1), targets

    def predict(self, signal):
        result = self.ml_algorithm.predict(self.transform_signal(signal)[np.newaxis, :])
        return str(self.label_encoder.inverse_transform(result[0]))

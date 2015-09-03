from matplotlib import lines
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pywt
from sklearn import cross_validation

from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from ui_oracle.transform_oracle import TransformOracle

__author__ = 'maeglin89273'


class LearningOracle:

    def __init__(self):
        self.wavelet = "coif4"
        self.transform_funcs = {"fft": self.fft_transform, "swt": self.swt_transform}

        plt.set_cmap("rainbow")

    def train(self, feature_selections, signals, targets):
        self.feature_selections = feature_selections
        signals, targets = self.preprocess(feature_selections, signals, targets)

        signals_train, signals_test, targets_train, targets_test = train_test_split(signals, targets, random_state=42)

        # self.ml_algorithm = KNeighborsClassifier()
        self.classifier = SVC(kernel="poly", degree=2)
        cv_score = cross_validation.cross_val_score(self.classifier, signals_train, targets_train, cv=5)

        self.classifier.fit(signals_train, targets_train)


        report = {}
        report["cv_score_mean"] = cv_score.mean()
        report["cv_score_std"] = cv_score.std()
        report["cv_folds"] = 5
        report["test_score"] = self.classifier.score(signals_test, targets_test)
        report["feature_selections"] = ", ".join((TransformOracle.TRANSFORMATIOIN_FULL_NAME_TABLE[init] for init in self.feature_selections))
        report["training_size"] = targets_train.size
        report["testing_size"] = targets_test.size
        report["learner_info"] = [("algorithm", "SVM"), ("kernel", "poly"), ("degree", 3)]

        return report

    def preprocess(self, feature_selections, signals, targets):
        self.transform_signals(feature_selections, signals)
        signals, targets = self.to_arrays(signals, targets)
        # signals = self.scale_signals(signals)

        return signals, targets

    def scale_signals(self, signals):
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(signals)


    def pca_plot3d(self, feature_selections, signals, targets):
        signals, targets = self.preprocess(feature_selections, signals, targets)

        rand_pca = RandomizedPCA(n_components=3)
        rand_pca.fit(signals)
        plot_values = rand_pca.transform(signals)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        collection = ax.scatter(plot_values[:, 0], plot_values[:, 1], plot_values[:, 2], c=targets)
        self.addLegend(ax, collection)
        plt.show()
        plt.close()

    def pca_plot2d(self, feature_selections, signals, targets):
        signals, targets = self.preprocess(feature_selections, signals, targets)
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


    def transform_signals(self, feature_selections, signals):
        for i, signal in enumerate(signals):
            signals[i] = self.transform_signal(feature_selections, signal)

        return signals

    def fft_transform(self, signal):
        transformed_signal = np.absolute(np.fft.rfft(signal))
        return transformed_signal[TransformOracle.FFT_START_FREQUENCY:len(transformed_signal) // 2]

    def dwt_transform(self, signal):
        dwt_result = pywt.wavedec(signal, self.wavelet, mode=pywt.MODES.per)
        return np.hstack(dwt_result[:-1])

    def swt_transform(self, signal):
        swt_result = pywt.swt(signal, "coif4", 3, 1)

        coeff_extracted = [swt_result[0][0]]
        for coeff_pair in swt_result:
            coeff_extracted.append(coeff_pair[1])

        return np.hstack(coeff_extracted)

    def transform_signal(self, feature_selections, signal):

        transformed_signal = []
        for tag in signal:
            for selection in feature_selections:
                transformed_signal.append(self.transform_funcs[selection](signal[tag]))

        return np.hstack(transformed_signal)

    def to_arrays(self, signals, targets):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(targets)
        targets = self.label_encoder.transform(targets)
        signals = np.array(signals)
        return signals, targets

    def predict(self, signal):
        # result = self.classifier.predict(self.scaler.transform(self.transform_signal(self.feature_selections, signal)[np.newaxis, :]))
        result = self.classifier.predict(self.transform_signal(self.feature_selections, signal)[np.newaxis, :])
        return str(self.label_encoder.inverse_transform(result[0]))

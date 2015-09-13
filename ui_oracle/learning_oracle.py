import traceback
from matplotlib import lines
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pywt
from sklearn import cross_validation

from sklearn.decomposition import RandomizedPCA, FastICA, PCA
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from ui_oracle.transform_oracle import TransformOracle

__author__ = 'maeglin89273'


class LearningOracle:

    def __init__(self):
        self.transform_oracle = TransformOracle()
        plt.set_cmap("rainbow")
        self.plot_table = {"2d": self.pca_plot2d, "3d": self.pca_plot3d}
        self.classifier_table = {"svm": SVC(), "k_nearest_neighbor": KNeighborsClassifier()}
        self.classifier_param_restructure_func = {"svm": self.svm_param_resturcture}
        self.scalers = {"mean_std_scaler": StandardScaler(), "min_max_scaler": MinMaxScaler()}

    def evaluate(self, training_settings, signals, targets):
        self.training_settings = training_settings
        signals, targets, transformations = self.preprocess(signals, targets)
        evaluation_settings = training_settings["evaluation"]

        if "train" in evaluation_settings:
           return self.train(signals, targets, transformations)
        elif "plot" in evaluation_settings:
            self.plot_table[evaluation_settings["plot"]["pca"]](signals, targets)

    def train(self, signals, targets, transformations):
        dataset_settings = self.training_settings["evaluation"]["train"]
        classifier_settings = self.training_settings["classifier"]
        test_size = dataset_settings["training_partition"] / 100
        signals_train, signals_test, targets_train, targets_test = train_test_split(signals, targets, train_size=test_size)

        classifier_key = LearningOracle.getOnlyKey(classifier_settings)
        self.classifier = self.classifier_table[classifier_key]

        params = classifier_settings[classifier_key]
        params = self.classifier_param_restructure_func[classifier_key](params)

        grid_search = GridSearchCV(self.classifier, params, cv=dataset_settings["cross_validation"])
        grid_search.fit(signals_train, targets_train)

        self.classifier = grid_search.best_estimator_
        self.classifier.fit(signals_train, targets_train)

        best_params_profile = sorted(grid_search.grid_scores_, key=lambda x: x.mean_validation_score, reverse=True)[0]

        report = {}
        report["Cross Validation Mean Score"] = "%.2f%%(+/-%.2f%%, %s folds)" % (best_params_profile.mean_validation_score * 100,
                                                                                  np.std(best_params_profile.cv_validation_scores) * 100,
                                                                                  dataset_settings["cross_validation"])

        if dataset_settings["evaluate_test_set"]:
            report["Test Set Score"] = "%.2f%%" % (self.classifier.score(signals_test, targets_test) * 100,)

        report["Size of Training Set"] = targets_train.size
        report["Size of Test Set"] = targets_test.size
        report["Best Parameters"] = best_params_profile.parameters

        return report

    @staticmethod
    def getOnlyKey(dict):
        return next(iter(dict.keys()))

    def svm_param_resturcture(self, params):

        new_params = []
        for kernel in params:
            single_settings = params[kernel]
            single_settings["kernel"] = [kernel]
            single_settings["C"] = single_settings["c"]
            del single_settings["c"]
            new_params.append(single_settings)

        return new_params

    def preprocess(self, signals, targets):
        feature_settings = self.training_settings["feature_extraction"]
        signals, transformations = self.transform_signals(feature_settings, signals)
        signals, targets = self.to_arrays(signals, targets)

        after_transformation_settings = feature_settings["after_transformation"]

        if "scaler" in after_transformation_settings:
            self.scaler = self.scalers[after_transformation_settings["scaler"]]
            signals = self.scaler.fit_transform(signals)

        if "pca" in after_transformation_settings:
            self.pca = RandomizedPCA(n_components=after_transformation_settings["pca"])
            signals = self.pca.fit_transform(signals)



        return signals, targets, transformations

    def pca_plot3d(self, signals, targets):

        rand_pca = RandomizedPCA(n_components=3)
        rand_pca.fit(signals)
        plot_values = rand_pca.transform(signals)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        collection = ax.scatter(plot_values[:, 0], plot_values[:, 1], plot_values[:, 2], c=targets)
        self.add_legend(ax, collection)
        plt.show()
        plt.close()

    def pca_plot2d(self, signals, targets):

        rand_pca = RandomizedPCA(n_components=2)
        rand_pca.fit(signals)
        plot_values = rand_pca.transform(signals)

        collection = plt.scatter(plot_values[:, 0], plot_values[:, 1], c=targets)

        self.add_legend(plt, collection)
        plt.show()
        plt.close()

    def add_legend(self, axis, collection):
        scatter_proxies = []
        labels = []
        for tag in self.label_encoder.classes_:
            scatter_proxies.append(lines.Line2D([0], [0], linestyle="none", marker="o",
                                                c=collection.to_rgba(self.label_encoder.transform(tag))))
            labels.append(tag)
        axis.legend(scatter_proxies, labels, numpoints=1)

    def transform_signals(self, feature_settings, signals):
        self.transform_funcs = []
        transformations = []
        if "wavelet_transform" in feature_settings:
            wavelet_settings = feature_settings["wavelet_transform"]
            self.transform_oracle.set_wt_type(wavelet_settings["type"])
            self.transform_oracle.set_wt_wavelet(wavelet_settings["wavelet"])
            range = wavelet_settings["level_range"]
            self.transform_oracle.set_wt_level_range(*range)
            self.transform_funcs.append(self.transform_oracle.wt_transform)
            transformations.append(wavelet_settings["type"] + "_wavelet_transform")

        if "fast_fourier_transform" in feature_settings:
            fft_settings = feature_settings["fast_fourier_transform"]
            self.transform_oracle.set_singal_size_and_sample_rate(feature_settings["window_size"], feature_settings["sample_rate"])
            self.transform_oracle.set_fft_freq_range(*fft_settings["frequency_range"])
            self.transform_funcs.append(self.transform_oracle.ranged_fft_transform)
            transformations.append("fast_fourier_transform")

        for i, signal in enumerate(signals):
            signals[i] = self.transform_signal(signal)

        return signals, transformations

    def transform_signal(self, signal):
        transformed_signal = []
        for tag in signal:
            for transform_func in self.transform_funcs:
                transformed_signal.append(transform_func(signal[tag]))

        return np.hstack(transformed_signal)

    def to_arrays(self, signals, targets):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(targets)
        targets = self.label_encoder.transform(targets)
        signals = np.array(signals)
        return signals, targets

    def predict(self, signal):
        # result = self.classifier.predict(self.scaler.transform(self.transform_signal(self.feature_selections, signal)[np.newaxis, :]))
        result = self.classifier.predict(self.transform_signal(signal)[np.newaxis, :])
        return str(self.label_encoder.inverse_transform(result[0]))

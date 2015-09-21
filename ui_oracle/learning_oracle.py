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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.utils import shuffle
from ui_oracle.transform_oracle import TransformOracle

__author__ = 'maeglin89273'


class LearningOracle:

    def __init__(self):
        self.transform_oracle = TransformOracle()
        plt.set_cmap("rainbow")
        self.training_settings = None
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

        signals_train = None
        targets_train = None
        if "training_partition" in dataset_settings:
            test_size = dataset_settings["training_partition"] / 100
            signals_train, signals_test, targets_train, targets_test = train_test_split(signals, targets, train_size=test_size)
        else:
            signals_train, targets_train = shuffle(signals, targets)
            signals_test = targets_test = np.array([])

        classifier_key = LearningOracle.get_only_key(classifier_settings)
        self.classifier = self.classifier_table[classifier_key]

        params = classifier_settings[classifier_key]
        params = self.classifier_param_restructure_func[classifier_key](params)

        cv_option = dataset_settings["cross_validation"]
        cv = None
        cv_text = None
        if "k_fold" in cv_option:
            cv = cross_validation.StratifiedKFold(targets_train, cv_option["k_fold"])
            cv_text = str(cv_option["k_fold"]) + " folds"
        else:
            cv = cross_validation.LeavePOut(targets_train.size, cv_option["leave_p_out"])
            cv_text = "leave " + str(cv_option["leave_p_out"]) + " out"

        grid_search = GridSearchCV(self.classifier, params, cv=cv)
        grid_search.fit(signals_train, targets_train)

        self.classifier = grid_search.best_estimator_
        self.classifier.fit(signals_train, targets_train)

        best_params_profile = sorted(grid_search.grid_scores_, key=lambda x: x.mean_validation_score, reverse=True)[0]

        report = {}
        report["Cross Validation Mean Score"] = "%.2f%%(+/-%.2f%%, %s)" % (best_params_profile.mean_validation_score * 100,
                                                                                  np.std(best_params_profile.cv_validation_scores) * 100,
                                                                                  cv_text)

        if dataset_settings["evaluate_test_set"]:
            report["Test Set Score"] = "%.2f%%" % (self.classifier.score(signals_test, targets_test) * 100,)

        report["Size of Training Set"] = targets_train.size
        report["Size of Test Set"] = targets_test.size
        report["Best Parameters"] = best_params_profile.parameters

        return report

    @staticmethod
    def get_only_key(dict):
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
        self.preprocess_steps = []
        if "scaler" in after_transformation_settings:
            self.preprocess_steps.append(self.scalers[after_transformation_settings["scaler"]])

        if "pca" in after_transformation_settings:
            self.preprocess_steps.append(RandomizedPCA(n_components=after_transformation_settings["pca"]))

        for step in self.preprocess_steps:
            signals = step.fit_transform(signals)

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
        if self.training_settings:
            result = self.classifier.predict(self.after_transformation(self.transform_signal(signal))[np.newaxis, :])
            return str(self.label_encoder.inverse_transform(result[0]))

        return None

    def after_transformation(self, signal):
        for step in self.preprocess_steps:
            signal = step.transform(signal)

        return signal

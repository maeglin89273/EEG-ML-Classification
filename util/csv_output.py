import csv
from nupic.algorithms.anomaly_likelihood import AnomalyLikelihood
from nupic.data.inference_shifter import InferenceShifter

__author__ = 'maeglin89273'

class ResultCSVOutput:
    def __init__(self):
        self.shifter = InferenceShifter()
        self.anomaly_likelihood_algorithm = AnomalyLikelihood(0,1)
        self.outFile = open("predict_output.csv", "wb")
        self.csvFile = csv.writer(self.outFile)
        self.csvFile.writerow(["actual_value", "predict_value", "anomaly_score", "anomaly_likelihood"])

    def write(self, index, fieldName, result, predictionStep=1):
        shifted_result = self.shifter.shift(result)
        inferenceValue = shifted_result.inferences['multiStepBestPredictions'][predictionStep]
        if inferenceValue:
            actualValue = shifted_result.rawInput[fieldName]
            anomalyScore = result.inferences['anomalyScore']
            likelihood = self.anomaly_likelihood_algorithm.anomalyProbability(actualValue, anomalyScore, index)
            self.csvFile.writerow([actualValue, inferenceValue, anomalyScore, likelihood])

    def finish(self):
        self.outFile.close()
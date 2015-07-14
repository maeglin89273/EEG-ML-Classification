import csv
import os
from nupic.frameworks.opf.modelfactory import ModelFactory
from swarm.sample_model_params import MODEL_PARAMS
from util.nupic_output import NuPICPlotOutput

__author__ = 'maeglin89273'

IS_TRAINING = False
TRAIN_DATA = ["rightstep_1/channel_test.csv"] * 3
PREDICT_DATA = "rightstep_1/channel_test.csv"
DATA_DIR = "../preprocessing/formatted_data/"


MODEL_DIR = os.path.join(os.path.abspath(os.getcwd()), "trained_model")

def getTestDataPath(dataFileName):
    return DATA_DIR + dataFileName


def skipHeader(csvFile):
    csvFile.next()
    csvFile.next()
    csvFile.next()


def predict(dataFileName, model):
    output = NuPICPlotOutput("output_simple_train", show_anomaly_score=True)

    model.disableLearning()

    with open(getTestDataPath(dataFileName), "rb") as input:
        csvFile = csv.reader(input)
        skipHeader(csvFile)
        for (timestamp, row) in enumerate(csvFile):
            value = float(row[0])
            result = model.run({"channel_value": value})
            output.write(timestamp, "channel_value", value, result, prediction_step=1)

def train(dataFileNames):
    model = ModelFactory.create(MODEL_PARAMS)
    model.enableInference({"predictedField": "channel_value"})
    for dataFileName in dataFileNames:
        with open(getTestDataPath(dataFileName), "rb") as input:
            csvFile = csv.reader(input)
            skipHeader(csvFile)

            print "training data %s" % dataFileName
            for row in csvFile:
                value = float(row[0])
                model.run({"channel_value": value})

    return model


if __name__ == "__main__":

    if IS_TRAINING:
        model = train(TRAIN_DATA)
        model.save(MODEL_DIR)
    else:
        model = ModelFactory.loadFromCheckpoint(MODEL_DIR)
        predict(PREDICT_DATA, model)


import csv
import os
from nupic.frameworks.opf.modelfactory import ModelFactory
from swarm.sample_model_params import MODEL_PARAMS
from util.csv_output import ResultCSVOutput
from util.nupic_output import NuPICPlotOutput

__author__ = 'maeglin89273'

IS_TRAINING = False
TRAIN_DATA = ["rightstep_1/channel_test.csv"] * 3
PREDICT_DATA = "step_1/channel_test.csv"
DATA_DIR = "../preprocessing/formatted_data/"
SLICED_TRAIN_DIR = "step_sliced/channel_3"

MODEL_DIR = os.path.join(os.path.abspath(os.getcwd()), "trained_model")

def getDataPath(dataFileName):
    return DATA_DIR + dataFileName


def skipHeader(csvFile):
    csvFile.next()
    csvFile.next()
    csvFile.next()


def predict(dataFileName, model):
    # output = NuPICPlotOutput("output_simple_train", show_anomaly_score=True)
    output = ResultCSVOutput()
    model.disableLearning()

    with open(getDataPath(dataFileName), "rb") as input:
        csvFile = csv.reader(input)
        skipHeader(csvFile)
        for (timestamp, row) in enumerate(csvFile):
            if timestamp % 1000 == 0:
                print "%s done" % timestamp
            value = float(row[0])
            result = model.run({"channel_value": value})
            # output.write(timestamp, "channel_value", value, result, prediction_step=1)
            output.write(timestamp, "channel_value", result, predictionStep=1)
    output.finish()

def train(dataFileNames):
    model = ModelFactory.create(MODEL_PARAMS)
    model.enableInference({"predictedField": "channel_value"})

    for dataFileName in dataFileNames:
        with open(getDataPath(dataFileName), "rb") as input:
            csvFile = csv.reader(input)
            skipHeader(csvFile)

            print "training data %s" % dataFileName
            for row in csvFile:
                value = float(row[0])
                model.run({"channel_value": value})

    return model


def getTrainFilePaths(trainFileDir):

    for (dirpath, dirnames, filenames) in os.walk(trainFileDir):
        return [os.path.join(dirpath, filename) for filename in filenames]

CYCLE = 2
def slicedTrain(trainFileDir):
    trainFileDir = getDataPath(trainFileDir)
    model = ModelFactory.create(MODEL_PARAMS)
    model.enableInference({"predictedField": "channel_value"})
    trainFiles = getTrainFilePaths(trainFileDir)
    for i in xrange(CYCLE):
        print "cycle %s:" % i
        for trainFile in trainFiles:
            with open(trainFile, "rb") as input:
                csvFile = csv.reader(input)
                skipHeader(csvFile)

                print "training data %s" % trainFile
                for row in csvFile:
                    value = float(row[0])
                    model.run({"channel_value": value})

            model.resetSequenceStates()
    return model

if __name__ == "__main__":

    if IS_TRAINING:
        model = slicedTrain(SLICED_TRAIN_DIR)
        model.save(MODEL_DIR)
    else:
        model = ModelFactory.loadFromCheckpoint(MODEL_DIR)
        predict(PREDICT_DATA, model)


import os
import pprint
from nupic.swarming import permutations_runner
import description

__author__ = 'maeglin89273'

MODEL_PARAMS_DIR_POSTFIX = "_model_params"

def getSampleName(sampleFileName):
    return os.path.splitext(sampleFileName)[0]
def modelParamsToString(modelParams):
    pp = pprint.PrettyPrinter(indent=1)
    return pp.pformat(modelParams)

def writeModelParamsToFile(sampleName, modelParams):
    cleanName = sampleName.replace(" ", "_").replace("-", "_")
    pythonFileName = "%s%s.py" % (cleanName, MODEL_PARAMS_DIR_POSTFIX)
    with open(pythonFileName, "w") as pyFile:
        modelParamsString = modelParamsToString(modelParams)
        pyFile.write("MODEL_PARAMS= \\\n%s" % modelParamsString)

    return pythonFileName

def runSwarm(sampleFileName, workers = 4):
    label = getSampleName(sampleFileName)
    modelParams = permutations_runner.runWithConfig(
        description.getDescription(sampleFileName), {"maxWorkers": workers, "overwrite": True},
        outDir =  label + MODEL_PARAMS_DIR_POSTFIX,
        outputLabel = label,
        verbosity=1
        )

    return modelParams

if __name__ == "__main__":
    sampleFileName = "sample.csv"
    modelParams = runSwarm(sampleFileName)
    outputFile = writeModelParamsToFile(getSampleName(sampleFileName), modelParams)

    print "swarm complete!"
    print "please import the module:"
    print "swarm.%s" % os.path.splitext(outputFile)[0]
__author__ = 'maeglin89273'

MAX_VALUE = 40.0
MIN_VALUE = -40.0

DESCRIPTION_TEMPLATE = {
  "includedFields": [
    {
      "fieldName": "channel_value",
      "fieldType": "float",
      "maxValue": MAX_VALUE,
      "minValue": MIN_VALUE
    }
  ],
  "streamDef": {
    "info": "data",
    "version": 1,
    "streams": [
      {
        "info": "data.csv",
        "source": "file://sample.csv",
        "columns": [
          "*"
        ]
      }
    ]
  },
  "inferenceType": "TemporalAnomaly",
  "inferenceArgs": {
    "predictionSteps": [
      1
    ],
    "predictedField": "channel_value"
  },
  "swarmSize": "medium"
}


def getDescription(sampleFilename):
    description = DESCRIPTION_TEMPLATE.copy()
    description["streamDef"]["streams"][0]["source"] = "file://" + sampleFilename
    return description

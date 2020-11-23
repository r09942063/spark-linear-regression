# Add Spark Python Files to Python Path
import sys
import os
SPARK_HOME = "/opt/bitnami/spark" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append( SPARK_HOME + "/python") # Add python files to Python Path


from pyspark.mllib.classification import LogisticRegressionWithSGD
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint

def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
         .setMaster("local") # run on local
         .setAppName("Logistic Regression") # Name of App
         .set("spark.executor.memory", "1g")) # Set 1 gig of memory
    sc = SparkContext(conf = conf) 
    return sc

def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """    
    feats = line.strip().split(",") 
    # labels must be at the beginning for LRSGD
    label = feats[len(feats) - 1] 
    feats = feats[: len(feats) - 1]
    features = [ float(feature) for feature in feats ] # need floats
    
    return LabeledPoint(label, features)

sc = getSparkContext()

# Load and parse the data
data = sc.textFile("/opt/bitnami/spark/data_banknote_authentication.txt")
parsedData = data.map(mapper)
# Train model
# model = LogisticRegressionWithSGD.train(parsedData)

def gradient(matrix, w):
    Y = matrix.labels   # point labels (first column of input file)
    X = matrix.features   # point coordinates
    # For each point (x, y), compute gradient function, then sum these up
    return ((1.0 / (1.0 + np.exp(- X.dot(w))) - Y ) * X.T)

def add(x, y):
    x += y
    return x
eta=1
iteration=100
for i in range(iterations):
    eta *=0.9
    w -= eta*parsedData.map(lambda m: gradient(m, w)).reduce(add) / float(parsedData.count())

    print("Final w: " + str(w))
# Predict the first elem will be actual data and the second 
# item will be the prediction of the model
def predict(f,w):
    Y=1.0/(1.0+np.exp(f*dot(w)))
    return round(Y)

labelsAndPreds = parsedData.map(lambda p: (p.label, predict(p.features,w)))

# Evaluating the model on training data
trainErr = labelsAndPreds.filter(lambda p: p[0]!=p[1]).count() / float(parsedData.count())

# Print some stuff
print("Training Error = " + str(trainErr))

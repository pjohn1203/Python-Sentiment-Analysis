#ML Asg 3

import csv
import math
import operator
import sys
import numpy
import nltk
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords  
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.model_selection import cross_val_score



# ------------------ Variable Names -------------------------


labelList = []
testList = []
vec = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1, max_features=20000)
vectorizedMatrix = []


# ------------------ Data Set Set-up Functions -----------------------



# Load and read our data set
def loadDataset():
 	trainSet = []
 	testSet = []
 	train = open('train.csv')
 	test = open('testset_1.csv')
 	trainData = list(csv.reader(train))
 	testData = list(csv.reader(test)) 	
 	for i in range(1, len(trainData)):
 		trainSet.append(trainData[i])

 	for i in range(1, len(testData)):
 		testSet.append(testData[i])


 	return trainSet, testSet


# iterates through dataSet and creates a list of all words
def wordListCreator(trainSet):
	wordList = []
	for i in range(0,len(trainSet)):
		instanceList = trainSet[i]
		instanceWords = instanceList[2]
		wordList.append(instanceWords)
	#print(wordList)
	return wordList

# function to create a list of labels given the DataSet

def getLabelList(trainSet):
	for i in range(0, len(trainSet)):
		trainInstance = trainSet[i]
		labelList.append(trainInstance[3])
	return labelList




# -------------- Cleaning Data Set Functions --------------------



# function to filter out characters that aren't words
def filterNonAlphabetical(instanceWords):
	return [w for w in wordList if w.isalpha()]


# function to remove stopwords from our list of words
def preProcessingStopWords(tokenList):
	stopWords = set(stopwords.words('english'))
	returnList = []
	for phrase in tokenList:
		words = phrase.split(' ')
		tempString = ""
		for word in words:
			if(word not in stopWords):
				tempString += word
				tempString += " "
		returnList.append(tempString)


	return returnList


# creates a dictionary with the word and its occurence in the dataset
# key = word & value = count
def convertToDict(processedList):
	d = {}
	for word in processedList:
		d[word] = d.get(word, 0) + 1
	return d





# --------------- Helper Functions for Testing -----------

# function to write our lists to a file.
def writeToFile(wordList, featureNames):

	w = open("wordList", 'w')
	f = open("featureList", 'w')

	for listitem in wordList:
		w.write('%s\n' % listitem)
	for listitem in featureNames:
		f.write('%s\n' % listitem)




# -------------- Main Functions --------------------------

def PreProcessing():
	#dataset in list form
	dataSet, testingSet = loadDataset()

	#List of all the phrases in DataSet
	wordList = wordListCreator(dataSet)
	testList = wordListCreator(testingSet)

	#List of all the Labels in the Data Set
	labelList = getLabelList(dataSet)

	#Cleaned word list of stop words
	processedList = preProcessingStopWords(wordList)


	#nltk feature extraction from the processed list of words
	# create "BoW" matrix fit to training data
	vectorizedExtraction = vec.fit_transform(processedList)
	vectorizedMatrix = vectorizedExtraction.toarray()
 
	#create "BoW" matrix to prior fit for the testing
	vectorizedTestingExtraction = vec.transform(testList)
	vectorizedTestingMatrix = vectorizedTestingExtraction.toarray()


	return vectorizedMatrix, labelList, vectorizedTestingMatrix



def trainingData(vectorizedMatrix, labelList):

	X = vectorizedMatrix
	y = labelList

	classifier2 = LogisticRegression(multi_class='auto')
	logreg = classifier2.fit(X,y)

	#cross validation accuracy score on data set.
	#scores = cross_val_score(classifier2, X, y, cv=5)
	#print(scores)

	return logreg



def predictTest(X, Y, classifier, dataset,X_test):
	openData = open(dataset)
	data = list(csv.reader(openData))
	data.pop(0) 
	phraseId = []
	for x in range(0,len(data)):
		instance = data[x]
		phraseId.append(instance[0])


	#w = open("predictions.csv" , "w")
	#w.write("PhraseId, Sentiment\n")
	predictedLabels = classifier.predict(X_test)

	return phraseId, predictedLabels


def makeCsv(phraseId, predictedLabels):
	data = list(csv.reader(open('testset_1.csv')))
	data.pop(0)
	w = open("predictions.csv" , "w")
	w.write("PhraseId, Sentiment\n")
	for i in range(0,len(predictedLabels)):
		w.write(data[i][0] + ',' + predictedLabels[i] + '\n')


# takes in the phrase id list and test set predictions
# and creates a scatter plot of the data
def createPlot(phraseIdList, testSetPredictions):
	xValues = []
	yValues = testSetPredictions
	fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

	for x in range(0, len(phraseIdList)):
		value = phraseIdList[x]
		if value not in xValues:
			xValues.append(value)

	xValues.sort()
	axs[0].bar(xValues, yValues)
	axs[1].scatter(xValues, yValues)
	axs[2].plot(xValues, yValues)
	fig.suptitle('Categorical Plotting')

	plt.show()




dataMatrix, labelList, testMatrix = PreProcessing()
logreg_classifier = trainingData(dataMatrix, labelList)
phraseIdList, testSetPredictions = predictTest(vectorizedMatrix,labelList,logreg_classifier,'testset_1.csv', testMatrix)
makeCsv(phraseIdList, testSetPredictions)
#createPlot(phraseIdList, testSetPredictions)




# ------------------ Ideas Box ---------------------------------

	# ----CLASSIFIER CODE-----

	#classifier3 = SVC(kernel='linear')
	#classifier3.fit(X_train,y_train)
	#makeCsv(X,y,classifier2,'train.csv',X_test)


	# training data split into test and train
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

	# checks accuracy against predicted labels and existing labels. 
	#prediction = classifier3.predict(X_test)
	#accuracy = accuracy_score(y, prediction)
	#print(accuracy)

	#Converted words in dataset to dictionary with total appearances as value
	#wordDictionary = convertToDict(processedList)

	# ----PREPROCESSING CODE-----

	#tfidf = TfidfTransformer(smooth_idf = False)
	#vectorizedTfidfMatrix = tfidf.fit_transform(vectorizedMatrix)

	#test = vectorizedTfidfMatrix.toarray()

	#j = open("tfidfMatrix", 'w')
	#for listitem in vectorizedMatrix:
	#	j.write('%s\n' % listitem)


	#vectorized list of features
	#featureNames = vec.get_feature_names()

	#writeToFile(wordList, featureNames)


	# ----PREPROCESSING CODE-----


	#instanceWords = re.sub(r'[^a-zA-Z ]', ' ', instanceWords)
	#tokenWords = nltk.word_tokenize(instanceWords)






	
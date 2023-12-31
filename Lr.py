import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import random
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

#load data
set1_human = "set1_human.json"
with open (set1_human, "r") as json_file:
    set1_human_data = json.load(json_file)

set2_human = "set2_human.json"
with open (set2_human, "r") as json_file:
    set2_human_data = json.load(json_file)
set1_machine = "set1_machine.json"
with open (set1_machine, "r") as json_file:
    set1_machine_data = json.load(json_file)
set2_machine = "set2_machine.json"
with open (set2_machine, "r") as json_file:
    set2_machine_data = json.load(json_file)
test = "test.json"
with open (test, "r") as json_file:
    test_data = json.load(json_file)

#pre-process
set1_human_data_txt = [i["txt"] for i in set1_human_data]
set1_machine_data_txt = [i["txt"] for i in set1_machine_data]

set2_human_data_txt = [i["txt"] for i in set2_human_data]
set2_machine_data_txt = [i["txt"] for i in set2_machine_data]

test_data_txt = [i["txt"] for i in test_data]
test_data_prompt = [i["prompt"] for i in test_data]

set1_human_data_prompt = [i["prompt"] for i in set1_human_data]
set1_machine_data_prompt = [i["prompt"] for i in set1_machine_data]

set2_human_data_prompt = [i["prompt"] for i in set2_human_data]
set2_machine_data_prompt = [i["prompt"] for i in set2_machine_data]


longestHDPIndex, longestHDP = max(enumerate(set1_human_data_prompt), key=lambda x: len(x[1]))
longestMDPIndex, longestMDP = max(enumerate(set1_machine_data_prompt), key=lambda x: len(x[1]))
longestHDTIndex, longestHDT = max(enumerate(set1_human_data_txt), key=lambda x: len(x[1]))
longestMDTIndex, longestMDT = max(enumerate(set1_machine_data_txt), key=lambda x: len(x[1]))

set1_human_data_txt.pop(longestHDTIndex)
set1_human_data_prompt.pop(longestHDTIndex)


longestHDTIndex, longestHDT = max(enumerate(set1_human_data_txt), key=lambda x: len(x[1]))
longestHDPIndex, longestHDP = max(enumerate(set1_human_data_prompt), key=lambda x: len(x[1]))
# #print(len(longestHDP)) #105
# #print("")
# #print(len(longestMDP)) #89


longestTPIndex, longestTP = max(enumerate(test_data_prompt), key=lambda x: len(x[1]))
longestTTIndex, longestTT = max(enumerate(test_data_txt), key=lambda x: len(x[1]))

# print(len(longestTP))  #190
# print(len(longestTT))  #1517
longestHDPIndex2, longestHDP2 = max(enumerate(set2_human_data_prompt), key=lambda x: len(x[1]))
longestMDPIndex2, longestMDP2 = max(enumerate(set2_machine_data_prompt), key=lambda x: len(x[1]))
longestHDTIndex2, longestHDT2 = max(enumerate(set2_human_data_txt), key=lambda x: len(x[1]))
longestMDTIndex2, longestMDT2 = max(enumerate(set2_machine_data_txt), key=lambda x: len(x[1]))


#print(len(longestHDP2)) 143
#print(len(longestMDP2)) 128
#print(len(longestHDT2)) 1496
#print(len(longestMDT2)) 1488

humanLabel = [1 for _ in range(len(set1_human_data_prompt))]

machineLabel = [0 for _ in range(len(set1_machine_data_prompt))]

humanLabel2 = [1 for _ in range(len(set2_human_data_prompt))]

machineLabel2 = [0 for _ in range(len(set2_machine_data_prompt))]

dataH = []
for i in range(len(set1_human_data_prompt)):
    subdata = []
    subdata.append(set1_human_data_prompt[i])
    subdata.append(set1_human_data_txt[i])
    subdata.append(humanLabel[i])
    result = tuple(subdata)
    dataH.append(result)

dataM = []
for i in range(len(set1_machine_data_prompt)):
    subdata = []
    subdata.append(set1_machine_data_prompt[i])
    subdata.append(set1_machine_data_txt[i])
    subdata.append(machineLabel[i])
    result = tuple(subdata)
    dataH.append(result)

# data = dataH+dataM
# random.shuffle(data)


dataH2 = []
for i in range(len(set2_human_data_prompt)):
    subdata = []
    subdata.append(set2_human_data_prompt[i])
    subdata.append(set2_human_data_txt[i])
    subdata.append(humanLabel2[i])
    result = tuple(subdata)
    dataH2.append(result)

dataM2 = []
for i in range(len(set2_machine_data_prompt)):
    subdata = []
    subdata.append(set2_machine_data_prompt[i])
    subdata.append(set2_machine_data_txt[i])
    subdata.append(machineLabel2[i])
    result = tuple(subdata)
    dataH2.append(result)

# data2 = dataH2+dataM2
# random.shuffle(data2)
data = dataH+dataM+dataH2+dataM2
random.shuffle(data)


trainPrompt = []
trainTxt = []
trainLabel = []
for value in data:
    trainPrompt.append(value[0])
    trainTxt.append(value[1])
    trainLabel.append(value[2])

testPrompt = test_data_prompt
testTxt = test_data_txt

#split data
x_train1, x_test1, y_train1, y_test1 = train_test_split(trainTxt, trainLabel, train_size = 0.8, random_state = 42)

x_train1ChangeToStr = []
for subList in x_train1:
    stc_str = " ".join([str(i) for i in subList])
    x_train1ChangeToStr.append(stc_str)

x_test1ChangeToStr = []
for subList in x_test1:
    stc_str = " ".join([str(value) for value in subList])
    x_test1ChangeToStr.append(stc_str)
    
testTxtChangeToStr = []
for subList in testTxt:
    stc_str = " ".join([str(i) for i in subList])
    testTxtChangeToStr.append(stc_str)

#bag of words
vectorizer = CountVectorizer()
vectorizer.fit(x_train1ChangeToStr)

x_train1tf = vectorizer.transform(x_train1ChangeToStr)
x_train1tf = x_train1tf.toarray()

x_test1tf = vectorizer.transform(x_test1ChangeToStr)
x_test1tf = x_test1tf.toarray()

testtf = vectorizer.transform(testTxtChangeToStr)
testtf = testtf.toarray()

y_test1 = np.array(y_test1)
y_train1 = np.array(y_train1)


#using smote
smote = SMOTE(sampling_strategy = 'auto',k_neighbors=16,random_state = 0)
x_trainSmote, y_train_smote = smote.fit_resample(x_train1tf,y_train1)

#hyper parameter tunning 
parameters= {'C': range(0,5,1)}
lr= LogisticRegression(penalty='l2',class_weight=None)
gridSearch = GridSearchCV(
    estimator = lr,
    param_grid = parameters,
    scoring = 'f1_macro',
    cv = 2,)
gridSearch.fit(x_trainSmote, y_train_smote)
theBestParameters = gridSearch.best_estimator_
print(theBestParameters)

#start the model
print("model start")
model = LogisticRegression(C=1/2.2,penalty='12',class_weight=None)
model.fit(x_trainSmote,y_train_smote)
pred = model.predict(x_test1tf)
print(classification_report(y_true = y_test1, y_pred =pred , digits = 6))

#finish the prediction and create a csv file.
preds = model.predict(testtf)
with open('predictions.csv', 'w') as f:
    f.write('Id,Predicted\n')
    for i, pred in enumerate(preds):
        f.write(f"{i},{pred}\n")        
print("done")
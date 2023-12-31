import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Bidirectional, LSTM, Concatenate
from keras.optimizers import Adam

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

#  0 and 1 represent machine and human labels, respectively
# 0 = machine
# 1 = human

#pre-process
set1_human_data_prompt = [i["prompt"] for i in set1_human_data]
set1_machine_data_prompt = [i["prompt"] for i in set1_machine_data]

set1_human_data_txt = [i["txt"] for i in set1_human_data]
set1_machine_data_txt = [i["txt"] for i in set1_machine_data]

test_data_prompt = [i["prompt"] for i in test_data]
test_data_txt = [i["txt"] for i in test_data]

set2_human_data_prompt = [i["prompt"] for i in set2_human_data]
set2_machine_data_prompt = [i["prompt"] for i in set2_machine_data]

set2_human_data_txt = [i["txt"] for i in set2_human_data]
set2_machine_data_txt = [i["txt"] for i in set2_machine_data]

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


for subList in set1_human_data_prompt:
    if len(subList) < len(longestTP):
        subList.extend([0]*(len(longestTP)-len(subList)))

for subList in set1_machine_data_prompt:
    if len(subList) < len(longestTP):
        subList.extend([0]*(len(longestTP)-len(subList)))

# print(len(longestHDT)) #2131
# print(len(longestMDT)) #2019
for subList in set1_human_data_txt:
    if len(subList) < len(longestHDT):
        subList.extend([0]*(len(longestHDT)-len(subList)))

for subList in set1_machine_data_txt:
    if len(subList) < len(longestHDT):
        subList.extend([0]*(len(longestHDT)-len(subList)))

for subList in set2_human_data_prompt:
    if len(subList) < len(longestTP):
        subList.extend([0]*(len(longestTP)-len(subList)))

for subList in set2_machine_data_prompt:
    if len(subList) < len(longestTP):
        subList.extend([0]*(len(longestTP)-len(subList)))
        
for subList in set2_human_data_txt:
    if len(subList) < len(longestHDT):
        subList.extend([0]*(len(longestHDT)-len(subList)))

for subList in set2_machine_data_txt:
    if len(subList) < len(longestHDT):
        subList.extend([0]*(len(longestHDT)-len(subList)))

#construct the labels
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

data = dataH+dataM
random.shuffle(data)

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

data2 = dataH2+dataM2
random.shuffle(data2)


for subList in test_data_prompt:
    if len(subList) < len(longestTP):
        subList.extend([0]*(len(longestTP)-len(subList)))

for subList in test_data_txt:
    if len(subList) < len(longestHDT):
        subList.extend([0]*(len(longestHDT)-len(subList)))




trainPrompt = []
trainTxt = []
trainLabel = []
for value in data:
    trainPrompt.append(value[0])
    trainTxt.append(value[1])
    trainLabel.append(value[2])

testPrompt = test_data_prompt
testTxt = test_data_txt



trainPromptArray = np.array(trainPrompt)
trainTxtArray = np.array(trainTxt)
trainLabelArray = np.array(trainLabel)
testPromptArray = np.array(testPrompt)
testTxtArray = np.array(testTxt)

trainPrompt2 = []
trainTxt2 = []
trainLabel2 = []
for value in data2:
    trainPrompt2.append(value[0])
    trainTxt2.append(value[1])
    trainLabel2.append(value[2])

trainPromptArray2 = np.array(trainPrompt2)
trainTxtArray2 = np.array(trainTxt2)
trainLabelArray2 = np.array(trainLabel2)

#BILSTM model
print("start model for set1")
timestepsTxt = 2131

inputShape = (timestepsTxt,)
inputs = Input(shape=inputShape)
embedding = Embedding(input_dim=5000, output_dim=15)(inputs)
bilstm = Bidirectional(LSTM(64))(embedding)
dense = Dense(64, activation='relu')(bilstm)
output = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=inputs, outputs=output)

#freeze the weights
bilstm.trainable = False

optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(trainTxtArray, trainLabelArray, epochs=10, batch_size=128)

print("start model for set2")
optimizer2 = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer2, loss='binary_crossentropy', metrics=['accuracy'])
model.fit( trainTxtArray2, trainLabelArray2, epochs=10, batch_size=64)


#finish the prediction and create a csv file.
predictions = model.predict(x=testTxtArray)
print("end predictions")
predictionsLabel =  [int(pred >= 0.5) for pred in predictions.flatten()]


with open('predictions', 'w') as f:
    f.write('Id,Predicted\n')
    for i, pred in enumerate(predictionsLabel):
        f.write(f"{i},{pred}\n")
print("done")
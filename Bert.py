from transformers import TFBertModel, BertTokenizer
import tensorflow as tf
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

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
set1_human_data_prompt = [i["prompt"] for i in set1_human_data]
set1_machine_data_prompt = [i["prompt"] for i in set1_machine_data]

set1_human_data_txt = [i["txt"] for i in set1_human_data]
set1_machine_data_txt = [i["txt"] for i in set1_machine_data]

test_data_prompt = [i["prompt"] for i in test_data]
test_data_txt = [i["txt"] for i in test_data]

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
# print(len(longestHDT)) #2131
# print(len(longestMDT)) #2019

#Token_ids [CLS](101), x,x,x,[SEP](102),[PAD], [PAD](0)
maxLength = 768

for subList in set1_human_data_prompt:
    subList.insert(0,101)
    subList.append(102)
    if len(subList) < maxLength:
        subList.extend([0]*(maxLength-len(subList)))
    else:
        subList = subList[:(maxLength-1)]
        subList.append(102)

for subList in set1_machine_data_prompt:
    subList.insert(0,101)
    subList.append(102)
    if len(subList) < maxLength:
        subList.extend([0]*(maxLength-len(subList)))
    else:
        subList = subList[:(maxLength-1)]
        subList.append(102)

for subList in set1_human_data_txt:
    subList.insert(0,101)
    subList.append(102)
    if len(subList) < maxLength:
        subList.extend([0]*(maxLength-len(subList)))
    else:
        subList = subList[:(maxLength-1)]
        subList.append(102)        


for subList in set1_machine_data_txt:
    subList.insert(0,101)
    subList.append(102)
    if len(subList) < maxLength:
        subList.extend([0]*(maxLength-len(subList)))
    else:
        subList = subList[:(maxLength-1)]
        subList.append(102)  

#Attention_mask
attention_mask_for_set1_human_data_prompt = []
for subList in set1_human_data_prompt:
    attention_mask = [1 if value != 0 else 0 for value in subList]
    attention_mask_for_set1_human_data_prompt.append(attention_mask)
    
attention_mask_for_set1_machine_data_prompt = []
for subList in set1_machine_data_prompt:
    attention_mask = [1 if value != 0 else 0 for value in subList]
    attention_mask_for_set1_machine_data_prompt.append(attention_mask)
    
attention_mask_for_set1_human_data_txt = []
for subList in set1_human_data_txt:
    attention_mask = [1 if value != 0 else 0 for value in subList]
    attention_mask_for_set1_human_data_txt.append(attention_mask)
    
attention_mask_set1_machine_data_txt = []
for subList in set1_machine_data_txt:
    attention_mask = [1 if value != 0 else 0 for value in subList]
    attention_mask_set1_machine_data_txt.append(attention_mask)

#token_type_ids 
token_type_ids_for_humanP = []
for i in range(len(set1_human_data_prompt)):
    subList = [0]*768
    token_type_ids_for_humanP.append(subList)
    
token_type_ids_for_humanT = []
for i in range(len(set1_human_data_txt)):
    subList = [0]*768
    token_type_ids_for_humanT.append(subList)   

    
token_type_ids_for_machineP = []
for i in range(len(set1_machine_data_prompt)):
    subList = [0]*768
    token_type_ids_for_machineP.append(subList)

token_type_ids_for_machineT = []
for i in range(len(set1_human_data_txt)):
    subList = [0]*768
    token_type_ids_for_machineT.append(subList)
    




humanLabel = [1 for _ in range(len(set1_human_data_prompt))]

machineLabel = [0 for _ in range(len(set1_machine_data_prompt))]

dataH = []
for i in range(len(set1_human_data_prompt)):
    subdata = []
    subdata.append(set1_human_data_prompt[i])
    subdata.append(attention_mask_for_set1_human_data_prompt[i])
    subdata.append(set1_human_data_txt[i])
    subdata.append(attention_mask_for_set1_human_data_txt[i])
    subdata.append(humanLabel[i])
    subdata.append(token_type_ids_for_humanP[i])
    subdata.append(token_type_ids_for_humanT[i])
    result = tuple(subdata)
    dataH.append(result)

dataM = []
for i in range(len(set1_machine_data_prompt)):
    subdata = []
    subdata.append(set1_machine_data_prompt[i])
    subdata.append(attention_mask_for_set1_machine_data_prompt[i])
    subdata.append(set1_machine_data_txt[i])
    subdata.append(attention_mask_set1_machine_data_txt[i])
    subdata.append(machineLabel[i])
    subdata.append(token_type_ids_for_machineP[i])
    subdata.append(token_type_ids_for_machineT[i])
    result = tuple(subdata)
    dataH.append(result)

data = dataH+dataM
random.shuffle(data)

trainPrompt = []
trainPromptAttentionMask = []
trainTxt = []
trainTxtAttentionMask = []
trainLabel = []
trainTypeP = []
trainTypeT = []
for value in data:
    trainPrompt.append(value[0])
    trainPromptAttentionMask.append(value[1])
    trainTxt.append(value[2])
    trainTxtAttentionMask.append(value[3])
    trainLabel.append(value[4])
    trainTypeP.append(value[5])
    trainTypeT.append(value[6])

trainTxtData = {}
trainTxtData['input_word_ids'] = trainTxt
trainTxtData['input_mask'] = trainTxtAttentionMask
trainTxtData['input_type_ids'] = trainTypeT

#model
bert_model_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
encoder = hub.KerasLayer(bert_model_url)

input_word_ids = tf.keras.layers.Input(shape=(768,), dtype=tf.int32, name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(768,), dtype=tf.int32, name="input_mask")
input_type_ids = tf.keras.layers.Input(shape=(768,), dtype=tf.int32, name="input_type_ids")
inputs = {
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}

embeddings = encoder(inputs)["pooled_output"]
output = tf.keras.layers.Dense(1, activation="sigmoid")(embeddings)
model = tf.keras.Model(inputs=inputs, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(trainTxtData, trainLabel,  epochs=10, batch_size=64)

# predictions = model.predict(x=testTxtArray)
# print("end predictions")
# predictionsLabel =  [int(pred >= 0.5) for pred in predictions.flatten()]


# with open('predictions', 'w') as f:
#     f.write('Id,Predicted\n')
#     for i, pred in enumerate(predictionsLabel):
#         f.write(f"{i},{pred}\n")
# print("done")
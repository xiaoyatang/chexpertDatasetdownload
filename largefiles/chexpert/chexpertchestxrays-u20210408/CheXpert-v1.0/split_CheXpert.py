
import json
from random import sample
import pandas as pd
Traindata_frt = pd.read_csv('/home/collab/u1368791/largefiles/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0/train.csv')

# %%
Traindata_frt = Traindata_frt.fillna(0)
Traindata_frt = Traindata_frt[Traindata_frt['Frontal/Lateral']=='Frontal'] #191027
Traindata_frt = Traindata_frt[:10000]


# %%
def applyZeor(x):
    # print(type(x))
    if type(x) == int or type(x) == float:
        if x <= 0.0:
            return 0
        else:
            return x
    else:
        return x
Traindata_frt= Traindata_frt.applymap(applyZeor)
Traindata_frt.insert(len(Traindata_frt.columns),'All Good',0)

# %%

# filtered_df = Train_front[Train_front['Pleural Effusion'].notnull()]#111776
# filtered_df.loc[filtered_df['Pleural Effusion']==-1.0, "Pleural Effusion"] =0. #change all -1.0 to 0(uncertain to 0)

# %%
column_names = Traindata_frt.columns.values.tolist()
label_name = column_names[5:]
print(label_name)

# %%
img_list = Traindata_frt['Path'].values.tolist() #111776
label = Traindata_frt.iloc[:, 5:].values.tolist()

# %%
dictionary = {}
# print(col_list[:5])
listImg = []
for j in list(zip(img_list,label)):
    indices = [i for i, x in enumerate(j[1]) if x == 1]
    if len(indices) > 0:
        labelName= label_name[sample(indices,1)[0]]
    else:
        # print(j[1])
        j[1][-1]=1
        # print(j[1])
        labelName = label_name[-1]
    # print(labelName)
    j = list(j)
    path,label=j
    path="/".join(path.split("/")[2:])
    # j.append(labelName)
    listImg.append([path,label,labelName])
# print(listImg)

dictionary = {
"train": listImg[:5000],
"val": listImg[5001:8001],
"test": listImg[8001:10001]
}

json_object = json.dumps(dictionary, indent=4)
with open("split_tang_CheXpert.json", "w") as outfile:
    outfile.write(json_object)
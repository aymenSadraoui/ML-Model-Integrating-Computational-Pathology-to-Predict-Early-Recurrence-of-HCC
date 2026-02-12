import os
import yaml
import torch
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

features_paths = config["paths"]["pth_to_features"]
models_paths = config["paths"]["pth_to_models"]
labels_path = 'data/Label_slides.xlsx'

# load data and labels
df0 = pd.read_excel(labels_path, sheet_name="PB").dropna(subset=["Patient"])
features_list = []
labels_list = []
# select patients with id < 90 for training
list_train_patients = [patient for patient in os.listdir(features_paths) if int(patient.split('_')[0][:-1]) < 90]
for feature_file in list_train_patients:
    if feature_file.endswith('_features.pt'):
        slide_id = feature_file.split('_')[0]
        feature_path = os.path.join(features_paths, feature_file)
        print(f"Slide ID: {slide_id}, Feature Path: {feature_path}")
        patient_id = int(slide_id[:-1])
        patient_list = df0["Patient"].tolist()
        if patient_id in patient_list:
            # load features from pt file
            features = torch.load(feature_path)['last_layer_embed']
            features_list.append(features.squeeze())
            # read label from excel file
            label = int(df0[df0["Patient"] == int(slide_id[:-1])]["Récidive avant 2 ans"].values[0]) 
            labels_list.append(label)

# convert to np array
features_array = torch.stack(features_list, dim=0).cpu().numpy()
labels_array = np.array(labels_list)

#same for internal validation
test_features_list = []
test_labels_list = []
# select patients with id > 90 for testing
list_train_patients = [patient for patient in os.listdir(features_paths) if int(patient.split('_')[0][:-1]) > 90]
for feature_file in list_train_patients:
    if feature_file.endswith('_features.pt'):
        slide_id = feature_file.split('_')[0]
        feature_path = os.path.join(features_paths, feature_file)
        print(f"Slide ID: {slide_id}, Feature Path: {feature_path}")
        patient_id = int(slide_id[:-1])
        patient_list = df0["Patient"].tolist()
        if patient_id in patient_list:
            # load features from pt file
            features = torch.load(feature_path)['last_layer_embed']
            test_features_list.append(features.squeeze())
            # read label from excel file
            label = int(df0[df0["Patient"] == int(slide_id[:-1])]["Récidive avant 2 ans"].values[0]) 
            test_labels_list.append(label)

# convert to np array
test_features_array = torch.stack(test_features_list, dim=0).cpu().numpy()
test_labels_array = np.array(test_labels_list)

#same for HM external validation
HM_test_features_list = []
HM_test_labels_list = []
df1 = pd.read_excel(labels_path, sheet_name="HMN").dropna(subset=["Patient"])
list_train_patients = os.listdir(features_paths)
for feature_file in os.listdir(features_paths):
    if feature_file.endswith('_features.pt'):
        slide_id = feature_file.split('_')[0]
        feature_path = os.path.join(features_paths, feature_file)
        print(f"Slide ID: {slide_id}, Feature Path: {feature_path}")
        patient_id = int(slide_id[:-1])
        patient_list = df1["Patient"].tolist()
        if patient_id in patient_list:
            # load features from pt file
            features = torch.load(feature_path)['last_layer_embed']
            HM_test_features_list.append(features.squeeze())
            # read label from excel file
            label = int(df1[df1["Patient"] == int(slide_id[:-1])]["Récidive avant 2 ans"].values[0]) 
            HM_test_labels_list.append(label)

# convert to np array
HM_test_features_array = torch.stack(HM_test_features_list, dim=0).cpu().numpy()
HM_test_labels_array = np.array(HM_test_labels_list)

# create a list of models 
models = []

# try a gaussian svm
svm = SVC()
models.append(svm)
# try a CatBoost
catboost = CatBoostClassifier()
#models.append(catboost)
# try a random forest
rf = RandomForestClassifier() 
models.append(rf)
# try a gradient boosting
gb = GradientBoostingClassifier() 
models.append(gb)
# try a adaboost
ada = AdaBoostClassifier() 
models.append(ada)
# try a xgboost
xgb = XGBClassifier()
models.append(xgb)
# try a decision tree
dt = DecisionTreeClassifier() 
models.append(dt)
# try a mlp
mlp = MLPClassifier() 
models.append(mlp)

results_dict = {}
for model in models:
    # cross validate the model
    F1_scores = cross_val_score(model, features_array, labels_array, cv=5,scoring='f1')
    accuracies = cross_val_score(model, features_array, labels_array, cv=5,scoring='accuracy')
    # compute mean and std of F1 scores and accuracies
    mean_f1 = np.mean(F1_scores)
    std_f1 = np.std(F1_scores) 
    mean_acc = np.mean(accuracies) 
    std_acc = np.std(accuracies) 
    print(f"Model: {model.__class__.__name__}, Mean F1 Score: {mean_f1:.4f} (+/- {std_f1:.4f}), Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")

    # test on test set
    model.fit(features_array, labels_array) 
    test_acc = model.score(test_features_array, test_labels_array) 
    HM_test_acc = model.score(HM_test_features_array, HM_test_labels_array) 
    test_pred = model.predict(test_features_array)
    HM_test_pred = model.predict(HM_test_features_array)
    test_f1 = f1_score(test_labels_array, test_pred) 
    HM_test_f1 = f1_score(HM_test_labels_array, HM_test_pred)
    print(f"Model: {model.__class__.__name__}, Test Accuracy: {test_acc:.4f}, F1: {test_f1}, HM Test Accuracy: {HM_test_acc:.4f}, F1: {HM_test_f1}")

    # add results to the dict 
    results_dict[model.__class__.__name__] = [mean_f1,std_f1,mean_acc, std_acc,test_acc, test_f1, HM_test_acc, HM_test_f1]


# save results to a csv file
results_df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['Mean F1 Score', 'Std F1 Score', 'Mean Accuracy', 'Std Accuracy','Test Accuracy', 'Test F1 Score', 'HM Test Accuracy', 'HM Test F1 Score']) 
results_df.to_csv('results/models/classifier_results.csv')



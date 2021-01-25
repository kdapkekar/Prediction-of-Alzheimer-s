import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,precision_score



sns.set()

df = pd.read_csv('oasis_longitudinal.csv')
df.head()

df = df.loc[df['Visit']==1] # use first visit data only because of the analysis we're doing
df = df.reset_index(drop=True) # reset index after filtering first visit data
df['M/F'] = df['M/F'].replace(['F','M'], [0,1]) # M/F column
df['Group'] = df['Group'].replace(['Converted'], ['Demented']) # Target variable
df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1,0]) # Target variable
df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1) # Drop unnecessary columns

def bar_chart(feature):
    Demented = df[df['Group']==1][feature].value_counts()
    Nondemented = df[df['Group']==0][feature].value_counts()
    df_bar = pd.DataFrame([Demented,Nondemented])
    df_bar.index = ['Demented','Nondemented']
    df_bar.plot(kind='bar',stacked=True, figsize=(8,5))
    
bar_chart('M/F')
plt.xlabel('Group')
plt.ylabel('Number of patients')
plt.legend()
plt.title('Gender and Demented rate')

#MMSE : Mini Mental State Examination
# Nondemented = 0, Demented =1
# Nondemented has higher test result ranging from 25 to 30. 
#Min 17 ,MAX 30
facet= sns.FacetGrid(df,hue="Group", aspect=3)
facet.map(sns.kdeplot,'MMSE',shade= True)
facet.set(xlim=(0, df['MMSE'].max()))
facet.add_legend()
plt.xlim(15.30)

#bar_chart('ASF') = Atlas Scaling Factor
facet= sns.FacetGrid(df,hue="Group", aspect=3)
facet.map(sns.kdeplot,'ASF',shade= True)
facet.set(xlim=(0, df['ASF'].max()))
facet.add_legend()
plt.xlim(0.5, 2)

#eTIV = Estimated Total Intracranial Volume
facet= sns.FacetGrid(df,hue="Group", aspect=3)
facet.map(sns.kdeplot,'eTIV',shade= True)
facet.set(xlim=(0, df['eTIV'].max()))
facet.add_legend()
plt.xlim(900, 2100)

#'nWBV' = Normalized Whole Brain Volume
# Nondemented = 0, Demented =1
facet= sns.FacetGrid(df,hue="Group", aspect=3)
facet.map(sns.kdeplot,'nWBV',shade= True)
facet.set(xlim=(0, df['nWBV'].max()))
facet.add_legend()
plt.xlim(0.6,0.9)

#AGE. Nondemented =0, Demented =0
facet= sns.FacetGrid(df,hue="Group", aspect=3)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, df['Age'].max()))
facet.add_legend()
plt.xlim(50,100)

#'EDUC' = Years of Education
# Nondemented = 0, Demented =1
facet= sns.FacetGrid(df,hue="Group", aspect=3)
facet.map(sns.kdeplot,'EDUC',shade= True)
facet.set(xlim=(df['EDUC'].min(), df['EDUC'].max()))
facet.add_legend()
plt.ylim(0, 0.16)

pd.isnull(df).sum() 

# Dropped the 8 rows with missing values in the column, SES
df_drop_null_values = df.dropna(axis=0, how='any')
pd.isnull(df_drop_null_values).sum()


df_drop_null_values['Group'].value_counts()

# Draw scatter plot between EDUC and SES
x = df['EDUC']
y = df['SES']

ses_not_null_index = y[~y.isnull()].index
x = x[ses_not_null_index]
y = y[ses_not_null_index]

# Draw trend line in red
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, y, 'go', x, p(x), "r--")
plt.xlabel('Education Level(EDUC)')
plt.ylabel('Social Economic Status(SES)')

plt.show()

df["SES"].fillna(df.groupby("EDUC")["SES"].transform("median"), inplace=True)

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import cross_val_score



# Dataset after dropping missing value rows
y = df_drop_null_values['Group'].values # Target for the model
x = df_drop_null_values[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']] # Features we use


# splitting into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.10, random_state=0)

# Feature scaling
feature_scaler = MinMaxScaler().fit(x_train)
x_train_scaled = feature_scaler.transform(x_train)
x_test_scaled = feature_scaler.transform(x_test)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc
acc = [] # list to store all performance metric

best_score=0
kfolds=5 # set the number of folds

for c in [0.001, 0.1, 1, 10, 100]:
    logRegModel = LogisticRegression(C=c)
    # perform cross-validation
    scores = cross_val_score(logRegModel, x_train_scaled, y_train, cv=kfolds, scoring='accuracy') # Get recall for each parameter setting
    
    # compute mean cross-validation accuracy
    score = np.mean(scores)
    
    # Find the best parameters and score
    if score > best_score:
        best_score = score
        best_parameters = c

# rebuild a model on the combined training and validation set
model = LogisticRegression(C=best_parameters).fit(x_train_scaled, y_train)

test_score = model.score(x_test_scaled, y_test)
y_pred = model.predict(x_test_scaled)
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
test_auc = auc(fpr, tpr)
print("Accuracy on validation set is:", best_score)
print("Accuracy with best C parameter is", test_score)
print("Test AUC is", test_auc)

plt.plot(fpr,tpr,label="Logistic Regression, auc="+str(test_auc))
plt.title('Model Performance')
plt.ylabel("FPR")
plt.xlabel("TPR")
plt.legend(loc=4)
plt.show()

#Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(x_train_scaled, y_train).predict(x_test_scaled)
accuracy_score(y_test,y_pred)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig, ax = plt.subplots()
ax.plot(y_test)
ax.plot(y_pred)


#Neural Network
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.regularizers import l2

model=Sequential()
model.add(Dense(100,activation='relu',input_dim=8,kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3,noise_shape=None,seed=None))

model.add(Dense(100,activation='relu',kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3,noise_shape=None,seed=None))

model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

model_output=model.fit(x_train_scaled,y_train,epochs=500,batch_size=20,verbose=1,validation_data=(x_test_scaled,y_test),)

print('Training Accuracy:',np.mean(model_output.history["accuracy"]))
print('Validation Accuracy:',np.mean(model_output.history["val_accuracy"]))

y_pred=model.predict(x_test_scaled)
rounded=[round(x[0]) for x in y_pred]
y_pred1=np.array(rounded,dtype='int64')
confusion_matrix(y_test,y_pred1)
precision_score(y_test,y_pred1)


# Plot training & validation accuracy values
plt.plot(model_output.history['accuracy'])
plt.plot(model_output.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(model_output.history['loss'])
plt.plot(model_output.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()




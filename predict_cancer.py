import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import f1_score,confusion_matrix
import os

classes = 2
batch = 8
epoch = 125

use_feature_selection = ['yes','no']

#read the data using pandas
dataframe = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath('data.csv')),'data.csv'))
#convert the values to numpy array
data = dataframe.values
#store the data in a separate variable to avoid confusion
x_data = data

#DATA CLEANING

#check for empty/NaN/Null values
print(dataframe.isnull())

#access the data values excluding labels
data = data[:,2:data.shape[1]]

#drop NaN column in the data
data = data[:,0:data.shape[1]-1]

#access labels
labels = x_data[:,1]
x_labels = labels

#convert the original labels to binary
for i in range(labels.shape[0]):
    if labels[i] == 'B':
        labels[i] = 1
    else:
        labels[i] = 0

def ffnn_model():
    # build a FFNN for classification
    model = Sequential()
    model.add(Dense(16,kernel_initializer='uniform',activation='relu',input_shape=(x_train.shape[1],)))
    model.add(Dense(16, kernel_initializer='uniform',activation='relu'))
    model.add(Dense(12,kernel_initializer='uniform',activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(classes-1,activation='sigmoid'))

    model.compile(optimizer=Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
    return model

#Analysis of data

#calculate the mean
mean = np.mean(data)
print('Mean=',mean)
#calculate variance of data
var = np.var(data)
print('Variance=',var)

#plot histogram of data
hist_data = np.reshape(data,(data.shape[0]*data.shape[1],1))

plt.hist(x=hist_data)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of breast cancer data')
plt.text(3000, 13100, r'$\mu=61.89,\ \sigma=52119.70$')
#plt.show()

#plot the heatmap of correlation between the features
sns.heatmap(dataframe.corr())
#plt.show()

#feature selection
def feature_selection(features_to_drop):
    X = data
    Y = labels
    Y = Y.astype('int')

    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X, Y)
    print(clf.feature_importances_)
    ind = np.argsort(clf.feature_importances_)

    #drop % of features from the original cleaned data
    amount = round(features_to_drop*X.shape[1])
    to_drop = X.shape[1] - amount
    ind = ind[to_drop:X.shape[1]]
    X = np.delete(X,ind,axis=1)
    print(X.shape)
    return (X,Y)

#Model for classification

#split 85% train and 15% test
train_amount = round(0.85*data.shape[0])
for i in range(len(use_feature_selection)):
    print('FS:',use_feature_selection[i])
    #with & without feature selection
    if use_feature_selection[i] == 'yes':
        #drop 50% of features
        (data,labels) = feature_selection(0.5)
        x_train = data[0:train_amount, :]
        y_train = labels[0:train_amount, ]

        x_test = data[train_amount + 1:data.shape[0], :]
        y_test = labels[train_amount + 1:data.shape[0], ]

        model = ffnn_model()
        history = model.fit(x_train, y_train, batch_size=batch, epochs=epoch, validation_data=(x_test, y_test))

        print(history.history.keys())
        acc = model.evaluate(x_test, y_test)
        y_pred = model.predict(x_test, batch_size=batch)

        y_pred = np.reshape(y_pred, (y_pred.shape[0],))

        f1 = f1_score(y_test.astype(float), y_pred.round())
        print('Test loss:', acc[0])
        print('Test accuracy:', acc[1])
        print('F1-score:', f1)
        c = confusion_matrix(y_test.astype(float), y_pred.round())

        sns.heatmap(c, annot=True, xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
        plt.title('Confusion matrix of samples for breast cancer classification')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        #plt.show()

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.show()

    else:
        x_train = data[0:train_amount,:]
        y_train = labels[0:train_amount,]

        x_test = data[train_amount+1:data.shape[0],:]
        y_test = labels[train_amount+1:data.shape[0],]

        model = ffnn_model()
        history = model.fit(x_train, y_train, batch_size=batch, epochs=epoch, validation_data=(x_test, y_test))

        acc = model.evaluate(x_test,y_test)
        y_pred = model.predict(x_test,batch_size=batch)

        y_pred = np.reshape(y_pred,(y_pred.shape[0],))

        f1 = f1_score(y_test.astype(float),y_pred.round())
        print('Test loss:',acc[0])
        print('Test accuracy:',acc[1])
        print('F1-score:',f1)
        c = confusion_matrix(y_test.astype(float),y_pred.round())

        sns.heatmap(c,annot=True,xticklabels=['Malignant','Benign'],yticklabels=['Malignant','Benign'])
        plt.title('Confusion matrix of samples for breast cancer classification')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        #plt.show()

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.show()

    #save the trained model used without feature selection
    if use_feature_selection[i] == 'yes':
        model_json = model.to_json()
        with open(os.path.join(os.path.dirname(os.path.abspath('model_with_fs.json')),'trained_models','model_with_fs.json'), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(os.path.join(os.path.dirname(os.path.abspath('model_with_fs.h5')),'trained_models','model_with_fs.h5'))
        print("Saved model to disk")
    #save trained model with feature selection
    else:
        model_json = model.to_json()
        with open(os.path.join(os.path.dirname(os.path.abspath('model_without_fs.json')),'trained_models','model_without_fs.json'), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(os.path.join(os.path.dirname(os.path.abspath('model_without_fs.h5')),'trained_models','model_without_fs.h5'))
        print("Saved model to disk")
    print('FS:', use_feature_selection[i],'Done!')


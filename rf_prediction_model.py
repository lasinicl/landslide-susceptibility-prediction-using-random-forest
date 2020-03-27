import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error

fields=['LandUse','LandForm','SoilTypeAndThickness','Geology','Slope','Aspect','SPI','TWI','STI','Rainfall','DistanceToWaterways','Class']

dataset = pd.read_csv(r'C:\Users\Dell\Desktop\FeaturesForAllPoints.csv',usecols=fields)

def encodeData(data):
    encode_data = preprocessing.LabelEncoder()    
    dataset['LandUse'] = encode_data.fit_transform(data.LandUse.astype(str))
    dataset['LandForm'] = encode_data.fit_transform(data.LandForm.astype(str))
    dataset['SoilTypeAndThickness'] = encode_data.fit_transform(data.SoilTypeAndThickness.astype(str))
    dataset['Geology'] = encode_data.fit_transform(data.Geology.astype(str))
    dataset['Slope'] = encode_data.fit_transform(data.Slope.astype(str))
    dataset['Aspect'] = encode_data.fit_transform(data.Aspect.astype(str))
    dataset['SPI'] = encode_data.fit_transform(data.SPI.astype(str))
    dataset['TWI'] = encode_data.fit_transform(data.TWI.astype(str))
    dataset['STI'] = encode_data.fit_transform(data.STI.astype(str))
    dataset['Rainfall'] = encode_data.fit_transform(data.Rainfall.astype(str))
    dataset['DistanceToWaterways'] = encode_data.fit_transform(data.DistanceToWaterways.astype(str))    
    data=data.fillna(-999)
    return data

dataset=encodeData(dataset)

X = dataset.iloc[:, 1:11].values
y=dataset.iloc[:,11].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1,shuffle='true')

#fit data to RF classifier
classifier=RandomForestClassifier(n_estimators=35,criterion='entropy',max_depth=30,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',class_weight='balanced',bootstrap='true',random_state=0,oob_score='true')
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

#generate confusion matrix
cm=confusion_matrix(y_test,y_pred)

#plot ROC curve
def plot_roc_cur(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

probs = classifier.predict_proba(X_test)
probs = probs[:, 1]
fper, tper, thresholds = roc_curve(y_test, probs) 
plot_roc_cur(fper, tper)

#plot Confusion Matrix
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax);

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['landslide', 'non-landslide'])
ax.yaxis.set_ticklabels(['landslide', 'non-landslide'])

#print performance measures
print("Accuracy on Training data: {:.2f}%".format(classifier.score(X_train,y_train)*100))
print("Accuracy on Test data: {:.2f}%".format(classifier.score(X_test, y_test)*100))
Sensitivity = float(cm[0,0])/float(cm[0,0]+cm[0,1])
print("Sensitivity:{:.2f}%".format(Sensitivity*100))
Specificity=float(cm[1,1])/float(cm[1,0]+cm[1,1])
print("Specificty:{:.2f}%".format(Specificity*100))
print("Mean Squared Error:{:.2f}%".format(mean_squared_error(y_test,y_pred)*100))
print("Kappa Index:{:.2f}".format(metrics.cohen_kappa_score(y_test,y_pred,weights='quadratic')))
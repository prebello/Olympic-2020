from __future__ import print_function, division

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

'''
RESULTS FOR BDT 
clemenciamora$ python3 -i jatos_BDT_anomaly.py 3
0.2
purity in test sample     : 41.07%
efficiency in test sample : 56.18%
accuracy in test sample   : 88.49%
>>> 
Marias-MacBook-Air-2:MestradoMatheusDNN clemenciamora$ python3 -i jatos_MLP_anomaly.py 2
0.3
purity in test sample     : 47.07%
efficiency in test sample : 53.15%
accuracy in test sample   : 90.78%
>>> 
Marias-MacBook-Air-2:MestradoMatheusDNN clemenciamora$ python3 -i jatos_MLP_anomaly.py 1
0.11
purity in test sample     : 14.22%
efficiency in test sample : 78.19%
accuracy in test sample   : 58.86%
>>> 
'''

jatos = pd.read_csv("jatos.csv")

plot_hists = False
xvars = 1 ## 1 only raw input, 2 only aggregate input, 3 all

import sys 
if len(sys.argv)>1:
    xvars= int(sys.argv[1])
if len(sys.argv)>2:
    plot_hists = bool(sys.argv[2].lower() =='true' or int(sys.argv[2])!=0 )
    
# sys.argv --> A lista de argumentos da linha de comando transmitida para um
# script Python. argv [0] é o nome do script, no caso ['jatos_BDT_anomaly.py']

def countnparts(row):
    n=0
    for i,val in enumerate(row):
        if i!=2101 and i%3==1 and val>0:
          n+=1
    return n

def sumpt(row):
    sum = 0
    for i,val in enumerate(row):
        if i!=2101 and i%3 ==1 and val>0:
            sum+=val
    return sum

def maxpt(row):
    pts = []
    for i,val in enumerate(row):
        if i!=2101 and i%3==1 and val>0:
            pts.append(val)
    ptnp = np.array(pts)
    return ptnp.max()


def countpt100(row):
    n = 0
    for i,val in enumerate(row):
        if i!=2101 and i%3==1 and val>=100.:
          n+=1
    return n


def countpt200(row):
    n = 0
    for i,val in enumerate(row):
        if i!=2101 and i%3==1 and val>=200.:
          n+=1
    return n
    

def countpt500(row):
    n = 0
    for i,val in enumerate(row):
        if i!=2101 and i%3==1 and val>=500.:
          n+=1
    return n

    
nparts = jatos.agg(countnparts,axis=1)
spt = jatos.agg(sumpt,axis=1)
mxpt = jatos.agg(maxpt,axis=1)
cpt100 = jatos.agg(countpt100,axis=1)
cpt200 = jatos.agg(countpt200,axis=1)
cpt500 = jatos.agg(countpt500,axis=1)


jatos['nparts'] = nparts
jatos['sum_pt'] = spt
jatos['max_pt'] = mxpt
jatos['n_pt100'] = cpt100
jatos['n_pt200'] = cpt200
jatos['n_pt500'] = cpt500


issig = (jatos['2100']==1)
sinal = jatos[issig]


isbg = (jatos['2100']==0)
ruido= jatos[isbg]


if plot_hists:
   plt.hist(jatos['nparts'],bins=100,range=(0,700),histtype="step")
   plt.hist(sinal['nparts'],bins=100,range=(0,700),histtype="step",label="sinal")
   plt.hist(ruido['nparts'],bins=100,range=(0,700),histtype="step",label="ruido")
   plt.legend(loc="upper right")
   plt.savefig('hist_jatos.png')


if xvars == 1:
    X = jatos.loc[:,'0':'2099']
    suffix = "onlypseudo"
elif xvars ==2 :
    X = jatos.loc[:,'nparts':'n_pt500']
    suffix = "onlyagg"
elif xvars ==3 :
    X = jatos.loc[:,'0':'2099']
    X['nparts']=jatos['nparts']
    X['sum_pt']=jatos['sum_pt']
    X['max_pt']=jatos['max_pt']
    X['n_pt100']=jatos['n_pt100']
    X['n_pt200']=jatos['n_pt200']
    X['n_pt500']=jatos['n_pt500']
    suffix = "allvars"
else:
    print("not such option")

y = jatos['2100']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)



  
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict
forest_clf = LGBMClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train, cv=5,
                                    method="predict_proba")

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class



from sklearn.metrics import roc_curve,precision_recall_curve
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train,y_scores_forest)
prec_forest, rec_forest, threshs_frst = precision_recall_curve(y_train, y_scores_forest)


plt.plot(rec_forest,prec_forest,"b:")
plt.xlabel("efficiency")
plt.ylabel("purity")
plt.title("Train")
plt.savefig("PRC_train"+suffix+".png")
plt.close()
#lt.show()

plt.plot(rec_forest,prec_forest*rec_forest,"g:")
plt.xlabel("efficiency")
plt.ylabel("efficiency*purity")
plt.savefig("RecPurity_train"+suffix+".png")
plt.close()
#plt.show()

forest_clf.fit(X_train,y_train)

y_pred_test_proba = forest_clf.predict_proba(X_test)
from sklearn.metrics import precision_score,recall_score,accuracy_score
y_test_score = y_pred_test_proba[:,1]
prec_test,rec_test, thresh_test = precision_recall_curve(y_test,y_test_score)

plt.plot(rec_test,prec_test,"b:")
plt.xlabel("recall")
plt.ylabel("purity")
plt.title("Test")
plt.savefig("PRC_test"+suffix+".png")
plt.close()
#plt.show()

plt.plot(rec_test,prec_test*rec_test,"g:")
plt.xlabel("recall")
plt.ylabel("recall*purity")
plt.title("Test")
plt.savefig("RecPurity_test"+suffix+".png")
plt.close()
#plt.show()


bidx = np.argmax(prec_forest*rec_forest)
best_cut = threshs_frst[bidx]
print(best_cut)

y_test_pred_best = y_test_score >= best_cut

print("purity in test sample     : {:2.2f}%".format(100*precision_score(y_test,y_test_pred_best)))
print("efficiency in test sample : {:2.2f}%".format(100*recall_score(y_test,y_test_pred_best)))
print("accuracy in test sample   : {:2.2f}%".format(100*accuracy_score(y_test,y_test_pred_best)))

hbg  = plt.hist(y_scores_forest[y_train==0],bins=100,range=(0,1),histtype='step',label='background')
hsig = plt.hist(y_scores_forest[y_train==1],bins=100,range=(0,1),histtype='step',label='signal')
uppery=np.max(hbg[0])*1.1
plt.plot([best_cut,best_cut],[0,uppery],"r:",label='best cut : {:2.2f}'.format(best_cut))
plt.axis([-0.01,1.01,0,uppery])
plt.xlabel("probability")
plt.ylabel("Number of events/bins of 0.01 width")
plt.title("Probability of signal for training sample Cross-validation")
plt.legend(loc="upper right")
plt.ylim(10e-2,5000)
plt.yscale('log')
plt.tight_layout()
plt.savefig("ProbTrain"+suffix+".pdf")
plt.close()
#plt.show()


hbgt =  plt.hist(y_test_score[y_test==0],bins=100,range=(0,1),histtype='step',label='background')
hsigt = plt.hist(y_test_score[y_test==1],bins=100,range=(0,1),histtype='step',label='signal')
uppery=np.max(hbgt[0])*1.1
plt.plot([best_cut,best_cut],[0,uppery],"r:",label='best cut : {:2.2f}'.format(best_cut))
plt.axis([-0.01,1.01,0,uppery])
plt.xlabel("probability")
plt.ylabel("Number of events/bins of 0.01 width")
plt.title("Probability of signal for test sample validation")
plt.ylim(10e-2,2500)
plt.yscale('log')
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("ProbTest"+suffix+".pdf")
plt.close()
#plt.show()



'''
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils


# Trainign batch size
batch_size = 64
# Number of training epochs
epochs = 50
# Fraction of the training data to be used as validation
val_split = 0.35
# Learning rate
lr=0.1

# Multilayer Perceptron model
model = Sequential()
model.add(Dense(input_dim=2101, activation="relu", units=500, kernel_initializer="normal"))
model.add(Dense(activation="relu", units=100, kernel_initializer="normal")) # after first layer no need to specify input_dim
model.add(Dense(activation="relu", units=50, kernel_initializer="normal")) # after first layer no need to specify input_dim
model.add(Dense(activation="sigmoid", units=1, kernel_initializer="normal"))
model.compile(optimizer=SGD(lr=lr), loss='mean_squared_error', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(X_train, y_train, validation_split=val_split, epochs=epochs, batch_size=batch_size, verbose=1)


# Evaluate
evaluation = model.evaluate(X_test, y_test, verbose=1)
print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))


# # Training History Visualization
#
# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper right')
# plt.show()
'''

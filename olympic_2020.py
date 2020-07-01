from __future__ import print_function, division
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import csv
import numpy as np
from pyjet import cluster,DTYPE_PTEPM
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import mean_squared_error
from numpy import array 
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer


'''
wc jatos.csv -> para as linhas do dataframe
less jatos.csv -> ler o dataframe no terminal
'''


fn = 'events_anomalydetection.h5'

df = pd.read_hdf(fn, stop = 11000)  # DataFrame do arquivo events_anomalydetection para 110000 eventos
#df['2100'] = df['2101']
#dtfr = df.to_csv('dataframe.csv')# salva o arquivo em um .csv
#df.head()

#print(df) # DataFrame completo 
#print(df.shape)
#print("Memory in GB:",sum(df.memory_usage(deep=True)) / (1024**3))

'''
um = dataframe[dataframe['2100'].isin([1.0])]  
zero = dataframe[dataframe['2100'].isin([0.0])]
'''

#df_olympic = df.to_csv('jatos.csv')  # Gerando um arquivo jatos.csv com os 11000 eventos

def foobar(jet):
    temp = []
    for elem in jet:
        temp.append([elem.pt, elem.eta, elem.phi, elem.mass])
    return(temp)
    ## Retorna [[pT, eta, phi, mass]] + [LABEL]

events_combined = df.T
print(np.shape(events_combined))

'''
leadpT = {}
leadphi = {}
alljets = {}
alljets_cu = {}
for mytype in ['background','signal']: #eliminar a linha 57 - 61
    leadpT[mytype]=[]
    leadphi[mytype] = []
    alljets[mytype]=[]
    alljets_cu[mytype] = []
    for i in range(11000): #len(events_combined)):
        if (i%1100==0):
            #print(mytype,i) #imprimi apenas o i e não o mytype
            pass
        issignal = events_combined[i][2100] #eliminar a linha 66 - 70
        if (mytype=='background' and issignal):
            continue
        elif (mytype=='signal' and issignal==0):
             continue
        pseudojets_input = np.zeros(len([x for x in events_combined[i][::3] if x > 0]), dtype=DTYPE_PTEPM)
        for j in range(700):
            if (events_combined[i][j*3]>0):
                pseudojets_input[j]['pT'] = events_combined[i][j*3]
                pseudojets_input[j]['eta'] = events_combined[i][j*3+1]
                pseudojets_input[j]['phi'] = events_combined[i][j*3+2]
                pass
            pass
        sequence = cluster(pseudojets_input, R=1.0, p=-1)
        jets = sequence.inclusive_jets(ptmin=20)
        leadpT[mytype] += [jets[0].pt]   #redifinir as linhas 81 - 84
        leadphi[mytype] +=[jets[0].phi]
        alljets_cu[mytype] += [jets] 
        alljets[mytype] += foobar(jets)
        pass



bg = alljets['background'] #equivalente a alljets['background']
sg = alljets['signal'] # equivalente a alljets['signal']
'''
nj_list = []
alljets_ra = []
alljets_po = []
for i in range(11000): #len(events_combined)):
    if (i%11000==0):
        #print(i) 
        pass
    pseudojets_input_po = np.zeros(len([x for x in events_combined[i][::3] if x > 0]), dtype=DTYPE_PTEPM)
    for j in range(700):
        if (events_combined[i][j*3]>0):
            pseudojets_input_po[j]['pT'] = events_combined[i][j*3]
            pseudojets_input_po[j]['eta'] = events_combined[i][j*3+1]
            pseudojets_input_po[j]['phi'] = events_combined[i][j*3+2]
            pass
        pass
    sequence_po = cluster(pseudojets_input_po, R=1.0, p=-1)
    jets_po = sequence_po.inclusive_jets(ptmin=20)
    alljets_po += [jets_po] 
    jets_features=[]
    for jet in jets_po:
        ptj = jet.pt
        etaj = jet.eta
        phij = jet.phi
        massj = jet.mass
        ncompj = len(jet)
        jets_features+=[ptj,etaj,phij,massj,ncompj]
    alljets_ra += [jets_features]
    nj = len(jets_po)
    nj_list.append(nj)
    pass
    #newdf_csv = newdf.to_csv('newdf_csv.csv')


new_df = pd.DataFrame(alljets_ra)
newdf = new_df.fillna(0) #preencher as colunas de NaN com 0
newdf['njets'] = nj_list
y = df[2100]
#newdf_csv = newdf.to_csv('newdf_csv.csv')

feature_prefix = ['pt','eta', 'phi', 'mass', 'count']
feature_names=[]
for i in range(5):
    for j in range(12):
        label = feature_prefix[i] + str(j+1)
        feature_names.append(label)
feature_names.append('njets')


X_train, X_test, y_train, y_test = train_test_split(newdf, y, test_size=0.2,random_state=42)
train = xgb.DMatrix(X_train,label=y_train,
                    missing=-999.0, feature_names=feature_names)
test = xgb.DMatrix(data=X_test,label=y_test,
                   missing=-999.0, feature_names=feature_names)
print('Number of training samples: {}'.format(train.num_row()))
print('Number of testing samples: {}'.format(test.num_row()))


param = []

# Boost5er parameters
param+= [('eta',        0.3)]               # learning rate
param+= [('max_depth',    9)]               # maximum depth of a tree
param+= [('subsample',  0.9)]               # fraction of events to train tree on
param+= [('colsample_bytree',0.5)]          # fraction of features to train tree on
# Learning task parameters
param+= [('objective', 'binary:logistic')]   # objective function
param+= [('eval_metric', 'error')]           # evaluation metric for cross validation
param+= [('eval_metric', 'logloss')] + [('eval_metric', 'rmse')]
print(param)
num_trees = 150  # number of trees to make

booster = xgb.train(param,train,num_boost_round=num_trees)

print(booster.eval(test))

predictions = booster.predict(test)


from sklearn.metrics import roc_curve,precision_recall_curve
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test,predictions)
prec_xgb, rec_xgb, threshs_xgb = precision_recall_curve(y_test, predictions)


plt.plot(rec_xgb,prec_xgb,"b:")
plt.xlabel("efficiency")
plt.ylabel("purity")
plt.title("Train for xgbgoost")
#plt.savefig("PRC_train"+suffix+".png")
plt.show()

plt.plot(rec_xgb,prec_xgb*rec_xgb,"g:")
plt.xlabel("efficiency")
plt.ylabel("efficiency*purity")
plt.title("Train for xgbgoost")
#plt.savefig("RecPurity_train"+suffix+".png")
plt.show()

#abaixo está o melhor corte para o xgb para separar o signal de Background
bidxg_xgb = np.argmax(prec_xgb*rec_xgb)
best_cut_xgb = threshs_xgb[bidxg_xgb]


y_test_pred_best_xgb = predictions >= best_cut_xgb

from sklearn.metrics import precision_score,recall_score,accuracy_score
print("purity in test sample for xgbgoost     : {:2.2f}%".format(100*precision_score(y_test,y_test_pred_best_xgb)))
print("efficiency in test sample for xgbgoost : {:2.2f}%".format(100*recall_score(y_test,y_test_pred_best_xgb)))
print("accuracy in test sample for xgbgoost   : {:2.2f}%".format(100*accuracy_score(y_test,y_test_pred_best_xgb)))


hbgt_xgb =  plt.hist(predictions[y_test==0],bins=100,range=(0,1),histtype='step',label='background')
hsigt_xgb = plt.hist(predictions[y_test==1],bins=100,range=(0,1),histtype='step',label='signal')
uppery_xgb=np.max(hbgt_xgb[0])*1.1
plt.plot([best_cut_xgb,best_cut_xgb],[0,uppery_xgb],"r:",label='best cut_grid')
plt.axis([-0.01,1.01,0,uppery_xgb])
plt.xlabel("probability")
plt.ylabel("Number of events/bins of 0.01 width")
plt.title("Probability of signal for test sample validation for xgbgoost")
plt.legend(loc="upper right")
plt.yscale('log')
plt.tight_layout()
plt.savefig('predictions_Xgboost.pdf')
plt.close()

from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score,precision_score,recall_score


def plotROC(predictions,y_test):
    # choose score cuts:
    cuts = np.linspace(0,1,500);
    n_truePos = np.zeros(len(cuts));
    n_falsePos = np.zeros(len(cuts));
    n_TotPos = len(np.where(y_test==1)[0])
    for i,cut in enumerate(cuts):
       y_pred = np.array([i>cut for i in predictions ])
       n_truePos[i] = len(np.where(predictions[y_test==1] > cut)[0]);
       n_falsePos[i] = len(np.where(predictions[y_test==0] > cut)[0]);
       if i%50 ==0:
         ascore = accuracy_score(y_test,y_pred)
         pscore = precision_score(y_test,y_pred)
         rscore = recall_score(y_test,y_pred)
         print("corte em {:2.1f} --> eficiência  {:2.1f} % e  pureza {:2.1f} %".format(cut,n_truePos[i]/n_TotPos *100,n_truePos[i]/(n_truePos[i]+n_falsePos[i])*100))
         print("                                                             accuracy_score = {:2.4f}     precision_score = {:2.4f}     recall_score = {:2.4f}\n".format(ascore,pscore,rscore))
    # plot efficiency vs. purity (ROC curve)
    plt.figure();

    custom_cmap3 = ListedColormap(['orange','yellow','lightgreen',"lightblue","violet"])
    plt.scatter((n_truePos/n_TotPos),n_truePos/(n_truePos + n_falsePos),c=cuts,cmap=custom_cmap3,label="ROC");
    # make the plot readable
    plt.xlabel('Efficiency',fontsize=12);
    plt.ylabel('Purity',fontsize=12);
    plt.colorbar()
    plt.show()

plotROC(predictions,y_test)


from sklearn.ensemble import RandomForestClassifier
'''
Os histogramas mostram picos com Probabilidade pŕoxima de 0,2 e 0,9, 
enquanto probabilidades próximas de 0 ou 1 são muito raras. 
Uma explicação para isso é dada por Niculescu-Mizil e Caruana 1: 
“Métodos como ensacamento e florestas aleatórias que mediam previsões 
de um conjunto básico de modelos podem ter dificuldade em fazer 
previsões próximas de 0 e 1 porque a variação nos modelos de base 
subjacentes influenciará as previsões. que deve estar perto de 
zero ou um desses valores. 
'''
from sklearn.model_selection import cross_val_predict
'''
Gere estimativas validadas cruzadamente para cada ponto de dados de entrada
Os dados são divididos de acordo com o parâmetro cv. 
Cada amostra pertence a exatamente um conjunto de testes 
e sua previsão é calculada com um estimador instalado no conjunto 
de treinamento correspondente. Passar essas previsões para uma métrica de avaliação 
pode não ser uma maneira válida de medir o desempenho da generalização.
Os resultados podem diferir de cross_validate e cross_val_score, 
a menos que todos os conjuntos de testes tenham o mesmo tamanho e a métrica se decomponha nas amostras.
'''

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train, cv=5,
                                    method="predict_proba")

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class



from sklearn.metrics import roc_curve,precision_recall_curve
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train,y_scores_forest)
prec_forest, rec_forest, threshs_frst = precision_recall_curve(y_train, y_scores_forest)


plt.plot(rec_forest,prec_forest,"b:")
plt.xlabel("efficiency")
plt.ylabel("purity")
plt.title("Train for RandomForest")
#plt.savefig("PRC_train"+suffix+".png")
plt.show()

plt.plot(rec_forest,prec_forest*rec_forest,"g:")
plt.xlabel("efficiency")
plt.ylabel("efficiency*purity")
plt.title("Train for RandomForest")
#plt.savefig("RecPurity_train"+suffix+".png")
plt.show()

forest_clf.fit(X_train,y_train)

y_pred_test_proba = forest_clf.predict_proba(X_test)

from sklearn.metrics import precision_score,recall_score,accuracy_score
y_test_score = y_pred_test_proba[:,1]
prec_test,rec_test, thresh_test = precision_recall_curve(y_test,y_test_score)

plt.plot(rec_test,prec_test,"b:")
plt.xlabel("recall")
plt.ylabel("purity")
plt.title("Test")
#plt.savefig("PRC_test"+suffix+".png")
plt.show()

plt.plot(rec_test,prec_test*rec_test,"g:")
plt.xlabel("recall")
plt.ylabel("recall*purity")
plt.title("Test")
#plt.savefig("RecPurity_test"+suffix+".png")
plt.show()

#
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
plt.plot([best_cut,best_cut],[0,uppery],"r:",label='best cut')
plt.axis([-0.01,1.01,0,uppery])
plt.xlabel("probability")
plt.ylabel("Number of events/bins of 0.01 width")
plt.title("Probability of signal for training sample Cross-validation")
plt.legend(loc="upper right")
#plt.savefig("ProbTrain"+suffix+".png")
plt.show()


hbgt =  plt.hist(y_test_score[y_test==0],bins=100,range=(0,1),histtype='step',label='background')
hsigt = plt.hist(y_test_score[y_test==1],bins=100,range=(0,1),histtype='step',label='signal')
uppery=np.max(hbgt[0])*1.1
plt.plot([best_cut,best_cut],[0,uppery],"r:",label='best cut')
plt.axis([-0.01,1.01,0,uppery])
plt.xlabel("probability")
plt.ylabel("Number of events/bins of 0.01 width")
plt.title("Probability of signal for test sample validation")
plt.legend(loc="upper right")
#plt.savefig("ProbTest"+suffix+".png")
plt.show()


from sklearn.model_selection import GridSearchCV 
'''
GridSearchCV  --> Pesquisa exaustiva sobre os valores de 
parâmetros especificados para um estimador.
Os parâmetros do estimador usado para aplicar 
esses métodos são otimizados pela pesquisa de 
grade validada cruzadamente sobre uma grade de parâmetros.

'''

params = {'n_estimators': [100, 150, 200], 'max_depth': list(range(4,11)), 'max_features': [0.25, 0.5, 0.75]}
grid_search_cv = GridSearchCV(RandomForestClassifier(random_state=42), params, n_jobs=-1, verbose=1, cv=3)

grid_search_cv.fit(X_train, y_train)

y_predproba_test = grid_search_cv.predict_proba(X_test)
y_test_score_grid = y_predproba_test[:,1]
precg_test,recg_test, threg_test = precision_recall_curve(y_test,y_test_score_grid)
#abaixo está a função para o melhor corte
bidxg = np.argmax(precg_test*recg_test)
best_cut_grid = threg_test[bidxg]


y_test_pred_best_grid = y_test_score_grid >= best_cut_grid

print("purity in test sample     : {:2.2f}%".format(100*precision_score(y_test,y_test_pred_best_grid)))
print("efficiency in test sample : {:2.2f}%".format(100*recall_score(y_test,y_test_pred_best_grid)))
print("accuracy in test sample   : {:2.2f}%".format(100*accuracy_score(y_test,y_test_pred_best_grid)))


hbgt =  plt.hist(y_test_score_grid[y_test==0],bins=100,range=(0,1),histtype='step',label='background')
hsigt = plt.hist(y_test_score_grid[y_test==1],bins=100,range=(0,1),histtype='step',label='signal')
uppery=np.max(hbgt[0])*1.1
plt.plot([best_cut_grid,best_cut_grid],[0,uppery],"r:",label='best cut_grid')
plt.axis([-0.01,1.01,0,uppery])
plt.xlabel("probability")
plt.ylabel("Number of events/bins of 0.01 width")
plt.title("Probability of signal for test sample validation given the grid")
plt.legend(loc="upper right")
#plt.savefig("ProbTest"+suffix+".png")
plt.show()

'''
mjj={}
for mytype in ['background','signal']:
    mjj[mytype]=[]
    for k in range(len(alljets_cu[mytype])):
        E = alljets_cu[mytype][k][0].e+alljets_cu[mytype][k][1].e
        px = alljets_cu[mytype][k][0].px+alljets_cu[mytype][k][1].px
        py = alljets_cu[mytype][k][0].py+alljets_cu[mytype][k][1].py
        pz = alljets_cu[mytype][k][0].pz+alljets_cu[mytype][k][1].pz
        mjj[mytype]+=[(E**2-px**2-py**2-pz**2)**0.5]
        pass
    pass

numero_sg = []
for i in range(982):
    numero_sg.append(len(alljets_cu['signal'][i]))
plt.hist(numero_sg)
plt.xlabel('numeros de jatos por eventos de signal')
plt.show()

numero_bg = []
for i in range(10018):
    numero_bg.append(len(alljets_cu['background'][i]))
plt.hist(numero_bg)
plt.xlabel('numeros de jatos por eventos de background')
plt.show()    


data_phi_sg = []
for i in range(982):
    delta_phi_sg = alljets_cu['signal'][i][0].phi - alljets_cu['signal'][i][1].phi  
    delta_phi_sg = abs(delta_phi_sg)
    data_phi_sg.append(delta_phi_sg)

data_phi_bg = []    
for i in range(10018):
    delta_phi_bg = alljets_cu['background'][i][0].phi - alljets_cu['background'][i][1].phi  
    delta_phi_bg = abs(delta_phi_bg)
    data_phi_bg.append(delta_phi_bg)


# histograma da difereça do angulo azimutal(delta phi)
# para os dois jatos mais energéticos de cada evento

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
n,b,p = plt.hist(data_phi_bg, bins=50, facecolor='r', alpha=0.2,label='background')
plt.hist(data_phi_sg, bins=b, facecolor='b', alpha=0.2,label='signal')
plt.xlabel(r'$\Delta\phi$')
plt.ylabel('Number of events')
plt.legend(loc='upper right')
plt.show()


data_pt_sg = []
for i in range(982):
    pt_sg = alljets_cu['signal'][i][0].pt
    data_pt_sg.append(pt_sg)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.hist(data_pt_sg, bins=50, facecolor='b', alpha=0.2,label='signal')
plt.xlabel(r'$pt$ -> mais energéticos')
plt.ylabel('Number of events')
plt.legend(loc='upper right')
plt.show()



data_pt_bg = []
for i in range(10018):
    pt_bg = alljets_cu['background'][i][0].pt
    data_pt_bg.append(pt_bg)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.hist(data_pt_bg, bins=50, facecolor='b', alpha=0.2,label='background')
plt.xlabel(r'$pt$ -> segundo mais energéticos')
plt.ylabel('Number of events')
plt.legend(loc='upper right')
plt.show()



fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
n,b,p = plt.hist(leadpT['background'], bins=50, facecolor='r', alpha=0.2,label='background')
plt.hist(leadpT['signal'], bins=b, facecolor='b', alpha=0.2,label='signal')
plt.xlabel(r'Leading jet $p_{T}$ [GeV]')
plt.ylabel('Number of events')
plt.legend(loc='upper right')
plt.show()
'''




from matplotlib.colors import ListedColormap
def plot_decision_boundary(clf, newdf, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, cut=0.4, contour=True):
    x1s = np.linspace(axes[0], axes[1], 200)
    x2s = np.linspace(axes[2], axes[3], 200)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    data_xnew = xgb.DMatrix(data=X_new,
                    missing=-999.0)
    y_pred_temp = clf.predict(data_xnew).reshape(x1.shape)
    y_pred = np.array([i>cut for i in y_pred_temp ])
    custom_cmap = ListedColormap(['orange','green','yellow'])
    plt.contourf(x1, x2, y_pred, alpha=0.5, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['violet','purple'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.9)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "ro", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    plt.show()

plot_decision_boundary(booster,X_test,y_test,axes=[-2,3,-2,3])    


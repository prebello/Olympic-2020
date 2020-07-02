from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
from pandas import set_option
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import fbeta_score, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from ml4jets_util import do_scaler, test_model
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
import pickle as pk
import imblearn
from sklearn.datasets import make_classification
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from imblearn.under_sampling import EditedNearestNeighbours
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectKBest,chi2,RFE,f_classif, mutual_info_classif
import seaborn as sns
from sklearn.metrics import precision_score,recall_score,accuracy_score,log_loss,roc_auc_score,average_precision_score,classification_report,f1_score
import mplhep as hep


DataSet_signal_exclusivo = pd.DataFrame(np.delete((np.array(pd.read_csv('amostra_exclusivo.csv'))),0,1))
DataSet_signal_semiexclusivo = pd.DataFrame(np.delete((np.array(pd.read_csv('amostra_semiesclusivo.csv'))),0,1))
DataSet_signal = pd.concat([DataSet_signal_exclusivo,DataSet_signal_semiexclusivo], axis=0)
DataSet_signal['target'] = 1

DataSet_backgraund = pd.DataFrame(np.delete((np.array(pd.read_csv('amostra_drellyan.csv'))),0,1))
DataSet_backgraund['target'] = 0
DataSet2 = pd.concat([DataSet_signal,DataSet_backgraund], ignore_index=0)

dataset = DataSet2.rename(columns={0:'Pt', 1:'MassaInvariante', 2:'Acoplanaridade', 3: 'DeltaEta', 4:'VerticePrimario', 5:'almir1', 6:'almir2', 7:'target'})

dataset_train, dataset_test_ = train_test_split(dataset, test_size = 0.30, random_state = 0)
TrainData, ValiData = train_test_split(dataset_train, test_size = 0.05, random_state = 0)

y_train_ = TrainData['target']
y_test_ = dataset_test_['target']
ValiLabel = ValiData['target']

TrainData = TrainData.drop(['target'], axis=1)
dataset_test_ = dataset_test_.drop(['target'], axis=1)
ValiData = ValiData.drop(['target'], axis=1)

train_data = lgb.Dataset(TrainData, label=y_train_)
test_data = lgb.Dataset(dataset_test_, label=y_test_)
vali_data = lgb.Dataset(ValiData, label = ValiLabel, reference = train_data)


evals_result = {}  # to record eval results for plotting

parameters = {
    'objective': "binary",
    'metric': 'auc',
    'boosting': "gbdt",
    'is_unbalance': "false",
    'num_leaves': 37,   
    'learning_rate': 0.0451,
    'max_depth': 69,
    'min_child_samples':40,
    'colsample_bytree': 0.9,
    'subsample':0.48,
    'n_estimators': 700
              }          

                 
model = lgb.train(
                  parameters,
                  train_data,
                  valid_sets=vali_data,
                  valid_names = ['Validation'],                 
                  num_boost_round=300,
                  early_stopping_rounds=200,
                  evals_result=evals_result
                  )

lgb.plot_metric(evals_result, title = 'Metric during training')
plt.savefig('/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/plot_metric.pdf')

pk.dump(model,open('train.pickle.dat_gbm_is_unbalance_SemPeso','wb'))

predictions = model.predict(dataset_test_)

from sklearn.metrics import roc_curve,precision_recall_curve,plot_precision_recall_curve,average_precision_score
fpr_lgb, tpr_lgb, thresholds_lgb = roc_curve(y_test_, predictions)
prec_lgb, rec_lgb, threshs_lgb = precision_recall_curve(y_test_, predictions)

def precision_recall(rec,prec,pwd):
    plt.plot(rec,prec,"b:")
    plt.xlabel("efficiency")
    plt.ylabel("purity")  
    plt.title("Precision Recall for lightgbm")
    plt.tight_layout()
    plt.savefig(pwd,format = 'pdf') # +suffix+
    plt.close()

precision_recall(rec_lgb,prec_lgb,"/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/PRC_train_gbm.pdf")

def precision_recall2(rec,prec,pwd):
    plt.plot(rec,prec*rec,"g:")
    plt.xlabel("efficiency")
    plt.ylabel("efficiency*purity")
    plt.title("Train for lightgbm")
    plt.tight_layout()
    plt.savefig(pwd,format = "pdf") # +suffix+
    plt.close()

precision_recall2(rec_lgb,prec_lgb,"/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/RecPurity_train_gbm.pdf")  


#abaixo está o melhor corte para o gbm para separar o signal de Background com PRECISION_RECALL_CURVE
def best_cut(prec,rec):
    bidxg_lgb = np.argmax(prec*rec)
    return threshs_lgb[bidxg_lgb] #MELHOR CORTE
   
best_cut_lgb1 = best_cut(prec_lgb,rec_lgb)
y_test_pred_best_lgb = predictions >= best_cut_lgb1


# Escala logarítimica
hbgt_lgb =  plt.hist(predictions[y_test_==0],bins=np.linspace(0,1,50), histtype='step',label='background')
hsigt_lgb = plt.hist(predictions[y_test_==1],bins=np.linspace(0,1,50), histtype='step',label='signal')
uppery_lgb=np.max(hsigt_lgb[0])*1.1
plt.plot([best_cut_lgb1,best_cut_lgb1],[0,uppery_lgb],"r:",label='best cut : {:2.2f}'.format(best_cut_lgb1))
plt.xlabel("Probability of signal for test sample for lightgbm")
plt.ylabel("Number of events/bins of 0.01 width in log scale")
plt.legend(loc="upper left")
plt.yscale('log')
plt.ylim(0,10e5)
plt.text(0.5,400000, "purity: {:2.2f}%".format(100*precision_score(y_test_,y_test_pred_best_lgb)),fontsize = 15)
plt.text(0.5,250000, "efficiency: {:2.2f}%".format(100*recall_score(y_test_,y_test_pred_best_lgb)),fontsize = 15)
plt.text(0.5,150000, "accuracy: {:2.2f}%".format(100*accuracy_score(y_test_,y_test_pred_best_lgb)), fontsize = 15)
plt.text(0.5,95000, "log loss: {:2.2f}%".format(100*log_loss(y_test_,y_test_pred_best_lgb)), fontsize = 15)
plt.text(0.5,65000, "ROC AUC: {:2.2f}%".format(100*roc_auc_score(y_test_,y_test_pred_best_lgb)), fontsize = 15)
plt.text(0.5,45000, "f1_score: {:2.2f}%".format(100*f1_score(y_test_,y_test_pred_best_lgb)), fontsize = 15)
plt.text(0.5,4000, 'SIGNAL REGION', color = 'red')
plt.style.use(hep.style.CMS)
plt.tight_layout()
plt.savefig('/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/hist_probTest_gbm_LogScale.pdf')
plt.close()

hbgt_lgb =  plt.hist(predictions[y_test_==0],bins=np.linspace(0,1,50), histtype='step',label='background')
hsigt_lgb = plt.hist(predictions[y_test_==1],bins=np.linspace(0,1,50),histtype='step',label='signal')
uppery_lgb=np.max(hbgt_lgb[0])*1.1
plt.plot([best_cut_lgb1,best_cut_lgb1],[0,uppery_lgb],"r:",label='best cut : {:2.2f}'.format(best_cut_lgb1))
plt.legend(loc="upper center")
plt.yscale('log')
plt.xlim(best_cut_lgb1 - 0.05, 1.01)
plt.text(0.5,5000, 'SIGNAL REGION', color = 'red', fontsize = 50)
plt.style.use(hep.style.CMS)
plt.tight_layout()
plt.savefig('/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/hist_probTest_gbm_LogScalecorte.pdf')
plt.close()

from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score,precision_score,recall_score

def plotROC(prediction,y_test, pwd):
    # choose score cuts:
    cuts = np.linspace(0,1,500);
    n_truePos = np.zeros(len(cuts));
    n_falsePos = np.zeros(len(cuts));
    n_TotPos = len(np.where(y_test==1)[0])
    for i,cut in enumerate(cuts):
       y_pred = np.array([i>cut for i in predictions])
       n_truePos[i] = len(np.where(predictions[y_test==1] > cut)[0]);
       n_falsePos[i] = len(np.where(predictions[y_test==0] > cut)[0]);
       if i%50 ==0:
         ascore = accuracy_score(y_test,y_pred)
         pscore = precision_score(y_test,y_pred)
         rscore = recall_score(y_test,y_pred)
         print("corte em {:2.1f} --> eficiência  {:2.1f} % e  pureza {:2.1f} %".format(cut,n_truePos[i]/n_TotPos *100,n_truePos[i]/(n_truePos[i]+n_falsePos[i])*100))
         print("accuracy_score = {:2.4f}     precision_score = {:2.4f}     recall_score = {:2.4f}\n".format(ascore,pscore,rscore))
    # plot efficiency vs. purity (ROC curve)
    plt.figure();
  
    custom_cmap3 = ListedColormap(['orange','yellow','lightgreen',"lightblue","violet"])
    plt.scatter((n_truePos/n_TotPos),n_truePos/(n_truePos + n_falsePos),c=cuts,cmap=custom_cmap3,label="ROC");
    # make the plot readable
    plt.xlabel('Efficiency',fontsize=12);
    plt.ylabel('Purity',fontsize=12);
    plt.tight_layout()
    plt.colorbar()
    plt.savefig(pwd,format = 'pdf')
    plt.close()

plotROC(predictions,y_test_, "/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/efficiency_x_purity_gbm.pdf")


def feature_importance(modelo, pwd):
    lgb.plot_importance(modelo,grid=True) # A "pontuação F" é o número de vezes que cada recurso é usado para dividir os dados em todas as árvores (vezes o peso dessa árvore).
    plt.xlim(0,5500)
    plt.tight_layout()
    plt.savefig(pwd, format = 'pdf')
    plt.close()

feature_importance(model,"/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/lightgbm_importance_gbm.pdf")

from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score,precision_score,recall_score

def ConfusionMatrix(y_test,y_predict,pwd):
    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_predict)
    print('Confusion matrix:\n', conf_mat)
    conf_mat = pd.DataFrame(conf_mat)
    conf_mat = conf_mat.rename(columns={0:'Drell-Yan', 1:'Signal'})
    conf_mat = conf_mat.T
    conf_mat = conf_mat.rename(columns={0:'Drell-Yan', 1:'Signal'})
    conf_mat = conf_mat.T

    sns.heatmap(conf_mat, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    plt.tight_layout()
    plt.savefig(pwd, format = 'pdf')
    plt.close()

ConfusionMatrix(y_test_,y_test_pred_best_lgb,'/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/confusion_matrix.pdf')


n_events_h_elastic = 200000
n_events_h_inel_el = 200000
n_events_h_dy = 48675378
total_lumi = 37.498268
limit_lumi = 18.516268
rap_gap_surv_elastic = 0.89
rap_gap_surv_inel_el = 0.76
rap_gap_surv_inelastic = 0.13
number_of_samples = 1200

scale_factor_elastic_x130 = limit_lumi*rap_gap_surv_elastic*0.017254036*1000 / n_events_h_elastic
scale_factor_inel_el_x130 = limit_lumi*rap_gap_surv_inel_el*0.025643500*1000 / n_events_h_inel_el 
scale_factor_dy = limit_lumi*5334000 / n_events_h_dy
scale_factor_dy_resample = limit_lumi*5334000 / ( n_events_h_dy*number_of_samples )

weight_signal_exclusivo = pd.DataFrame([scale_factor_elastic_x130]*len(DataSet_signal_exclusivo), columns = ['pesos'])
weight_signal_semiexclu = pd.DataFrame([scale_factor_inel_el_x130]*len(DataSet_signal_semiexclusivo), columns = ['pesos'])
weight_backgr = pd.DataFrame([scale_factor_dy]*len(DataSet_backgraund), columns = ['pesos'])
weight_signal = pd.concat([weight_signal_exclusivo,weight_signal_semiexclu], axis = 0)

pesos = pd.concat([weight_signal,weight_backgr], axis = 0)

##############################################################################################################################

# USANDO O CLASSIFICADOR LGBMClassifeir

##############################################################################################################################

y_train_ = np.array(y_train_)
y_test_ = np.array(y_test_)
ValiLabel = np.array(ValiLabel)

from sklearn.model_selection import cross_val_predict

clf_LGBM = LGBMClassifier(
                    boosting_type='gbdt', 
                    objective='binary', 
                    num_leaves = 37,   
                    learning_rate = 0.0451,
                    max_depth = 69,
                    min_child_samples = 40,
                    colsample_bytree = 0.9,
                    subsample = 0.48,
                    n_estimators = 700,
                    metric = 'auc'                       
                    )
from sklearn.model_selection import cross_val_predict

# PARA O TRAIN

y_probas_LGBM = cross_val_predict(clf_LGBM, TrainData, y_train_, cv=5, method="predict_proba")
y_probas_LGBM = y_probas_LGBM[:,1]
fpr_lgb_classif_TRAIN, tpr_lgb_classif_TRAIN, thresholds_lgb_classif_TRAIN = roc_curve(y_train_, y_probas_LGBM)
prec_lgb_classif_TRAIN, rec_lgb_classif_TRAIN, threshs_lgb_classif_TRAIN = precision_recall_curve(y_train_, y_probas_LGBM)

precision_recall(rec_lgb_classif_TRAIN,prec_lgb_classif_TRAIN,"/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/PRC_train_gbmCLASSIFEIR_TRAIN.pdf")
precision_recall2(rec_lgb_classif_TRAIN,prec_lgb_classif_TRAIN,"/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/RecPurity_train_gbmCLASSIFEIR_TRAIN.pdf")  

# PARA O TEST

clf_LGBM.fit(TrainData, y_train_)
predict_proba_LGBM_TEST = clf_LGBM.predict_proba(dataset_test_)
class_prediction_LGBM = clf_LGBM.predict(dataset_test_)
predict_proba_LGBM_TEST = predict_proba_LGBM_TEST[:,1]

fpr_lgb_classif_TEST, tpr_lgb_classif_TEST, thresholds_lgb_classif_TEST = roc_curve(y_test_, predict_proba_LGBM_TEST)
prec_lgb_classif_TEST, rec_lgb_classif_TEST, threshs_lgb_classif_TEST = precision_recall_curve(y_test_, predict_proba_LGBM_TEST)

precision_recall(prec_lgb_classif_TEST,prec_lgb_classif_TEST,"/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/PRC_train_gbmCLASSIFEIR_TEST.pdf")
precision_recall2(prec_lgb_classif_TEST,prec_lgb_classif_TEST,"/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/RecPurity_train_gbmCLASSIFEIR_TEST.pdf")  

bidx = np.argmax(prec_lgb_classif_TRAIN*rec_lgb_classif_TRAIN)
best_cut_train = threshs_lgb_classif_TRAIN[bidx]
print(best_cut_train)
y_test_pred_best_train = y_probas_LGBM >= best_cut_train

hbgt_lgb =  plt.hist(y_probas_LGBM[y_train_==0],bins=np.linspace(0,1,50), histtype='step',label='background')
hsigt_lgb = plt.hist(y_probas_LGBM[y_train_==1],bins=np.linspace(0,1,50), histtype='step',label='signal')
uppery_lgb=np.max(hsigt_lgb[0])*1.1
plt.plot([best_cut_train,best_cut_train],[0,uppery_lgb],"r:",label='best cut : {:2.2f}'.format(best_cut_train))
plt.xlabel("probability")
plt.ylabel("Number of events/bins of 0.01 width")
plt.title("Training sample Cross-validation")
plt.legend(loc="upper left")
plt.yscale('log')
plt.ylim(0,10e5)
plt.text(0.5,400000, "purity: {:2.2f}%".format(100*precision_score(y_train_,y_test_pred_best_train)),fontsize = 15)
plt.text(0.5,250000, "efficiency: {:2.2f}%".format(100*recall_score(y_train_,y_test_pred_best_train)),fontsize = 15)
plt.text(0.5,150000, "accuracy: {:2.2f}%".format(100*accuracy_score(y_train_,y_test_pred_best_train)), fontsize = 15)
plt.text(0.5,95000, "log loss: {:2.2f}%".format(100*log_loss(y_train_,y_test_pred_best_train)), fontsize = 15)
plt.text(0.5,65000, "ROC AUC: {:2.2f}%".format(100*roc_auc_score(y_train_,y_test_pred_best_train)), fontsize = 15)
plt.text(0.5,45000, "f1_score: {:2.2f}%".format(100*f1_score(y_train_,y_test_pred_best_train)), fontsize = 15)
plt.text(best_cut_train+0.03,4000, 'SIGNAL REGION', color = 'red')
plt.style.use(hep.style.ROOT)
plt.tight_layout()
plt.savefig('/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/hist_probTest_gbm_LogScaleCLASSIFEIR_TRAIN.pdf')
plt.close()

bidx = np.argmax(prec_lgb_classif_TEST*rec_lgb_classif_TEST)
best_cut_test = threshs_lgb_classif_TEST[bidx]
print(best_cut_test)
y_test_pred_best_test = predict_proba_LGBM_TEST >= best_cut_test

hbgt_lgb =  plt.hist(predict_proba_LGBM_TEST[y_test_==0],bins=np.linspace(0,1,50), histtype='step',label='background')
hsigt_lgb = plt.hist(predict_proba_LGBM_TEST[y_test_==1],bins=np.linspace(0,1,50), histtype='step',label='signal')
uppery_lgb=np.max(hsigt_lgb[0])*1.1
plt.plot([best_cut_test,best_cut_test],[0,uppery_lgb],"r:",label='best cut : {:2.2f}'.format(best_cut_test))
plt.xlabel("probability")
plt.ylabel("Number of events/bins of 0.01 width")
plt.title("Test sample validation")
plt.legend(loc="upper left")
plt.yscale('log')
plt.ylim(0,10e5)
plt.text(0.5,400000, "purity: {:2.2f}%".format(100*precision_score(y_test_,class_prediction_LGBM)),fontsize = 15)
plt.text(0.5,250000, "efficiency: {:2.2f}%".format(100*recall_score(y_test_,class_prediction_LGBM)),fontsize = 15)
plt.text(0.5,150000, "accuracy: {:2.2f}%".format(100*accuracy_score(y_test_,class_prediction_LGBM)), fontsize = 15)
plt.text(0.5,95000, "log loss: {:2.2f}%".format(100*log_loss(y_test_,class_prediction_LGBM)), fontsize = 15)
plt.text(0.5,65000, "ROC AUC: {:2.2f}%".format(100*roc_auc_score(y_test_,class_prediction_LGBM)), fontsize = 15)
plt.text(0.5,45000, "f1_score: {:2.2f}%".format(100*f1_score(y_test_,class_prediction_LGBM)), fontsize = 15)
plt.text(best_cut_test+0.03,4000, 'SIGNAL REGION', color = 'red')
plt.style.use(hep.style.ROOT)
plt.tight_layout()
plt.savefig('/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/hist_probTest_gbm_LogScaleCLASSIFEIR_TEST.pdf')
plt.close()


feature_importance(clf_LGBM,"/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/lightgbm_importance_gbmCLASSIFEIR.pdf")
ConfusionMatrix(y_test_,class_prediction_LGBM,'/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/confusion_matrixCLASSIFEIR.pdf')


from mlxtend.plotting import plot_decision_regions,plot_learning_curves

plot_learning_curves(np.array(TrainData), np.array(y_train_), np.array(dataset_test_), y_test_, clf = clf_LGBM, print_model = False)
plt.style.use(hep.style.ROOT)
plt.tight_layout()
plt.savefig('/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/plot_learning_curvesCLASSIFEIR.pdf')
plt.close()

y_train_ = np.array(y_train_)

X = np.array(TrainData[['Pt','DeltaEta']])
clf_LGBM.fit(X,y_train_)
contourf_kwargs = {'alpha': 0.9}
scatter_kwargs = {'alpha': 0.2}
plot_decision_regions(X,y_train_, clf=clf_LGBM, hide_spines = True, contourf_kwargs = contourf_kwargs, scatter_kwargs = scatter_kwargs)
plt.xlabel('$PT_{\mu^{+}\mu^{-}}$')
plt.ylabel('$|\Delta \eta|$')
plt.title('Training Sample for $PT_{\mu^{+}\mu^{-}} X \Delta \eta$')
plt.xlim(0,500)
plt.tight_layout()
plt.savefig('/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/DecisionBoundary_PTxDeltaEta.png')
plt.close()

X = np.array(TrainData[['MassaInvariante','DeltaEta']])
clf_LGBM.fit(X,y_train_)
contourf_kwargs = {'alpha': 0.9}
scatter_kwargs = {'alpha': 0.2}
plot_decision_regions(X,y_train_, clf=clf_LGBM, hide_spines = True, contourf_kwargs = contourf_kwargs, scatter_kwargs = scatter_kwargs)
plt.xlabel('$\mathcal{M}_{\mu^{+}\mu^{-}}$')
plt.ylabel('$\Delta \eta$')
plt.title('Training Sample for $\mathcal{M}_{\mu^{+}\mu^{-}}$  x  $\Delta \eta$')
plt.xlim(0,1100)
plt.tight_layout()
plt.savefig('/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/DecisionBoundary_MassaInvarixDeltaEta.png')
plt.close()

X = np.array(TrainData[['Acoplanaridade','DeltaEta']])
clf_LGBM.fit(X,y_train_)
contourf_kwargs = {'alpha': 0.9}
scatter_kwargs = {'alpha': 0.2}
plot_decision_regions(X,y_train_, clf=clf_LGBM, hide_spines = True, contourf_kwargs = contourf_kwargs, scatter_kwargs = scatter_kwargs)
plt.xlabel('Acoplanarity')
plt.ylabel('$\Delta \eta$')
plt.title('Training Sample for Acoplanarity X $\Delta \eta$')
plt.xlim(-0.2,1.6)
plt.tight_layout()
plt.savefig('/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/DecisionBoundary_AcoplanarityxDeltaEta.png')
plt.close()

X = np.array(TrainData[['almir1','DeltaEta']])
clf_LGBM.fit(X,y_train_)
contourf_kwargs = {'alpha': 0.9}
scatter_kwargs = {'alpha': 0.2}
plot_decision_regions(X,y_train_, clf=clf_LGBM, hide_spines = True, contourf_kwargs = contourf_kwargs, scatter_kwargs = scatter_kwargs)
plt.xlabel('Almir1')
plt.ylabel('$\Delta \eta$')
plt.title('Training Sample for almir1 x $\Delta \eta$')
plt.xlim(0,25)
plt.tight_layout()
plt.savefig('/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/DecisionBoundary_Almir1xDeltaEta.png')
plt.close()

X = np.array(TrainData[['Pt','MassaInvariante']])
clf_LGBM.fit(X,y_train_)
contourf_kwargs = {'alpha': 1}
scatter_kwargs = {'alpha': 0.2}
plot_decision_regions(X,y_train_, clf=clf_LGBM, hide_spines = True, contourf_kwargs = contourf_kwargs, scatter_kwargs = scatter_kwargs)
plt.xlabel('$PT_{\mu^{+}\mu^{-}}$')
plt.ylabel('$\mathcal{M}_{\mu^{+}\mu^{-}}$')
plt.title('Training Sample for $PT_{\mu^{+}\mu^{-}}$ x $\mathcal{M}_{\mu^{+}\mu^{-}} $')
plt.xlim(0,500)
plt.tight_layout()
plt.savefig('/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/DecisionBoundary_PtxMassaInvari.png')
plt.close()

X = np.array(TrainData[['Pt','Acoplanaridade']])
clf_LGBM.fit(X,y_train_)
contourf_kwargs = {'alpha': 0.9}
scatter_kwargs = {'alpha': 0.2}
plot_decision_regions(X,y_train_, clf=clf_LGBM, hide_spines = True, contourf_kwargs = contourf_kwargs, scatter_kwargs = scatter_kwargs)
plt.xlabel('$PT_{\mu^{+}\mu^{-}}$')
plt.ylabel('Acoplanarity')
plt.title('Training Sample for $PT_{\mu^{+}\mu^{-}}$ x  Acoplanarity')
plt.xlim(0,600)
plt.ylim(-0.2,1.2)
plt.tight_layout()
plt.savefig('/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/DecisionBoundary_PtxAcoplanaridade.png')
plt.close()

X = np.array(TrainData[['Pt','almir1']])
clf_LGBM.fit(X,y_train_)
contourf_kwargs = {'alpha': 0.9}
scatter_kwargs = {'alpha': 0.2}
plot_decision_regions(X,y_train_, clf=clf_LGBM, hide_spines = True, contourf_kwargs = contourf_kwargs, scatter_kwargs = scatter_kwargs)
plt.xlabel('$PT_{\mu^{+}\mu^{-}}$')
plt.ylabel('almir1')
plt.title('Training Sample for $PT_{\mu^{+}\mu^{-}}$ x almir1')
plt.xlim(0,180)
plt.tight_layout()
plt.savefig('/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/DecisionBoundary_Ptxalmir1.png')
plt.close()

X = np.array(TrainData[['MassaInvariante','almir1']])
clf_LGBM.fit(X,y_train_)
contourf_kwargs = {'alpha': 0.9}
scatter_kwargs = {'alpha': 0.2}
plot_decision_regions(X,y_train_, clf=clf_LGBM, hide_spines = True, contourf_kwargs = contourf_kwargs, scatter_kwargs = scatter_kwargs)
plt.xlabel('$\mathcal{M}_{\mu^{+}\mu^{-}}$')
plt.ylabel('almir1')
plt.title('Training Sample for $\mathcal{M}_{\mu^{+}\mu^{-}}$ x almir1')
plt.xlim(0,800)
plt.tight_layout()
plt.savefig('/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/DecisionBoundary_MassaInvarixalmir1.png')
plt.close()

X = np.array(TrainData[['MassaInvariante','Acoplanaridade']])
clf_LGBM.fit(X,y_train_)
contourf_kwargs = {'alpha': 0.9}
scatter_kwargs = {'alpha': 0.2}
plot_decision_regions(X,y_train_, clf=clf_LGBM, hide_spines = True, contourf_kwargs = contourf_kwargs, scatter_kwargs = scatter_kwargs)
plt.xlabel('$\mathcal{M}_{\mu^{+}\mu^{-}}$')
plt.ylabel('Acoplanarity')
plt.title('Training Sample for $\mathcal{M}_{\mu^{+}\mu^{-}} x $ Acoplanarity')
plt.xlim(0,800)
plt.tight_layout()
plt.savefig('/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/DecisionBoundary_MassaInvarixAcoplanaridade.png')
plt.close()

X = np.array(TrainData[['almir1','Acoplanaridade']])
clf_LGBM.fit(X,y_train_)
contourf_kwargs = {'alpha': 0.9}
scatter_kwargs = {'alpha': 0.2}
plot_decision_regions(X,y_train_, clf=clf_LGBM, hide_spines = True, contourf_kwargs = contourf_kwargs, scatter_kwargs = scatter_kwargs)
plt.xlabel('almir1')
plt.ylabel('Acoplanarity')
plt.title('Training Sample for almir1 x  Acoplanarity')
plt.ylim(-0.2,1.2)
plt.tight_layout()
plt.savefig('/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/DecisionBoundary_Almir1xAcoplanaridade.png')
plt.close()

X = np.array(TrainData[['VerticePrimario','Acoplanaridade']])
clf_LGBM.fit(X,y_train_)
contourf_kwargs = {'alpha': 0.9}
scatter_kwargs = {'alpha': 0.2}
plot_decision_regions(X,y_train_, clf=clf_LGBM, hide_spines = True, contourf_kwargs = contourf_kwargs, scatter_kwargs = scatter_kwargs)
plt.xlabel('VerticePrimario')
plt.ylabel('Acoplanarity')
plt.title('Training Sample for VerticePrimario x  Acoplanarity')
#plt.ylim(-0.2,1.2)
plt.tight_layout()
plt.savefig('/home/matheus/nTuplas/algoritmos/graficos/graficos_lightgbm/DecisionBoundary_VerticePrimarioxAcoplanaridade.png')
plt.close()

# Retreinando o modelo novamente com as duas features mais importantes 
# Os dois modelos derão o DeltaEta e o PT




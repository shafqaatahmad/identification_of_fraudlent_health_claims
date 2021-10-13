# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 19:30:07 2021

@author: shafqaat.ahmad
"""
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from collections import Counter
import os 
from datetime import date
from sklearn.feature_selection import chi2
from scipy import stats
import seaborn as sns
import matplotlib.pylab as plt
from numpy import percentile
from sklearn.feature_selection import SelectKBest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,LabelEncoder
import matplotlib.cm as cm
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor



path="E:\Python WD"
os.chdir(path) 
claimsdf = pd.read_csv("TrainDataCombined.csv",low_memory='False') # Loading the file

claimsdf.dtypes

#Converting data type to datetime
claimsdf.DOB=pd.to_datetime(claimsdf.DOB)
claimsdf.ClaimStartDt=pd.to_datetime(claimsdf.ClaimStartDt)
claimsdf.ClaimEndDt=pd.to_datetime(claimsdf.ClaimEndDt)

#Adding column Age and assiging values
claimsdf['Age']=claimsdf['ClaimEndDt'].dt.year-claimsdf['DOB'].dt.year

#Adding column ClaimDurationDays and assiging values
claimsdf['ClaimDurationDays']=claimsdf['ClaimEndDt']-claimsdf['ClaimStartDt']
claimsdf['ClaimDurationDays']=claimsdf['ClaimDurationDays'].dt.days

# Assign 1 if DOD is not null 0 otherwise
#claimsdf['Dead_YN']=claimsdf['DOD'].apply(lambda x: 1 if claimsdf['DOD'].isna else 0  ) 

# Gender replace
claimsdf['Gender'] = claimsdf['Gender'].replace(2,0)

# RenalDiseaseIndicator replace Y with 1
claimsdf['RenalDiseaseIndicator']=claimsdf['RenalDiseaseIndicator'].replace('Y',1)

###START---Replace 2 with 0 ** This is Frank's logic and his idea, I copied from his code**#######
claimsdf['ChronicCond_Alzheimer'] = claimsdf['ChronicCond_Alzheimer'].replace(2,0)
claimsdf['ChronicCond_Heartfailure'] = claimsdf['ChronicCond_Heartfailure'].replace(2,0)
claimsdf['ChronicCond_KidneyDisease'] = claimsdf['ChronicCond_KidneyDisease'].replace(2,0)
claimsdf['ChronicCond_Cancer'] = claimsdf['ChronicCond_Cancer'].replace(2,0)
claimsdf['ChronicCond_ObstrPulmonary'] = claimsdf['ChronicCond_ObstrPulmonary'].replace(2,0)
claimsdf['ChronicCond_Depression'] = claimsdf['ChronicCond_Depression'].replace(2,0)
claimsdf['ChronicCond_Diabetes'] = claimsdf['ChronicCond_Diabetes'].replace(2,0)
claimsdf['ChronicCond_IschemicHeart'] = claimsdf['ChronicCond_IschemicHeart'].replace(2,0)
claimsdf['ChronicCond_Osteoporasis'] = claimsdf['ChronicCond_Osteoporasis'].replace(2,0)
claimsdf['ChronicCond_rheumatoidarthritis'] = claimsdf['ChronicCond_rheumatoidarthritis'].replace(2,0)
claimsdf['ChronicCond_stroke'] = claimsdf['ChronicCond_stroke'].replace(2,0)
###END----Replace 2 with 0 ** This is Frank's logic and his idea, i copied from his code**#######

# There is no race "4" in, so replacing 5 with 4
claimsdf['Race'] = claimsdf['Race'].replace(5,4)

#Updating No with 0 and Yes with 1 
claimsdf['PotentialFraud']=claimsdf['PotentialFraud'].replace("No",0)
claimsdf['PotentialFraud']=claimsdf['PotentialFraud'].replace("Yes",1)

## Pick columns ** Drop unecessary columns and columns where null account for more than 30% of total values##

claimsdf=claimsdf[['Age','Gender','Race','RenalDiseaseIndicator','State','NoOfMonths_PartACov',
'NoOfMonths_PartBCov','ChronicCond_Alzheimer','ChronicCond_Heartfailure','ChronicCond_KidneyDisease',
'ChronicCond_Cancer','ChronicCond_ObstrPulmonary','ChronicCond_Depression','ChronicCond_Diabetes',
'ChronicCond_IschemicHeart','ChronicCond_Osteoporasis','ChronicCond_rheumatoidarthritis',
'ChronicCond_stroke','IPAnnualReimbursementAmt','IPAnnualDeductibleAmt','OPAnnualReimbursementAmt',
'OPAnnualDeductibleAmt','ClaimDurationDays','InscClaimAmtReimbursed','AttendingPhysician','DeductibleAmtPaid',
'ClmDiagnosisCode_1','ClmDiagnosisCode_2','PotentialFraud']]

## Drop Duplicates and NAs
claimsdf.isna().sum()
claimsdf=claimsdf.dropna()
claimsdf.isna().sum().sum()

claimsdf.head()


## Finally we get 29 columns

######### Label encoder###############


label_encoder = preprocessing.LabelEncoder()

claimsdf.AttendingPhysician=label_encoder.fit_transform(claimsdf.AttendingPhysician)
claimsdf.ClmDiagnosisCode_1=label_encoder.fit_transform(claimsdf.ClmDiagnosisCode_1)
claimsdf.ClmDiagnosisCode_2=label_encoder.fit_transform(claimsdf.ClmDiagnosisCode_2)
#########################################################################

claimsdf.to_csv("Claims_cleaned_with_labelencoding.csv")


################## Perform CLustering #######################################
#############################################################################


#claimscluster1=claimsdf.drop(['PotentialFraud'],axis=1)


claimscluster1=claimsdf[['Age','Gender','Race','RenalDiseaseIndicator','State','NoOfMonths_PartACov',
'NoOfMonths_PartBCov','OPAnnualReimbursementAmt','OPAnnualDeductibleAmt','ClaimDurationDays'
,'InscClaimAmtReimbursed','DeductibleAmtPaid','AttendingPhysician','ClmDiagnosisCode_1','PotentialFraud']]

labelencoder=LabelEncoder()

claimscluster1['ClmDiagnosisCode_1']=labelencoder.fit_transform(claimscluster1['ClmDiagnosisCode_1'])
claimscluster1['AttendingPhysician']=labelencoder.fit_transform(claimscluster1['AttendingPhysician'])


claimscluster1=claimsdf[['State','ClaimDurationDays','PotentialFraud']]

claimscluster1.astype(int64)

#Scaling the data
data=StandardScaler().fit_transform(claimscluster1)

##########################Number of clusters################
sse = {}
for k in range(1, 19):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    sse[k] = kmeans.inertia_ # SSE to closest cluster centroid
plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()
############################################################

model = KMeans(n_clusters=8, init='k-means++', max_iter=100, n_init=1, verbose=1)
cluster = model.fit_predict((data))
cluster.shape
clusterdata=claimscluster1
clusterdata['Cluster'] = cluster
#clusterdata.shape
#display(clusterdata.sort_values(by='Cluster'))


#'Age','Gender','Race','RenalDiseaseIndicator','State','NoOfMon#ths_PartACov',
#'NoOfMonths_PartBCov','OPAnnualReimbursementAmt','OPAnnualDeductibleAmt',
#'ClaimDurationDays','InscClaimAmtReimbursed','DeductibleAmtPaid','PotentialFraud'

zcol=claimscluster1.columns.get_loc('AttendingPhysician')
icol=claimscluster1.columns.get_loc('ClaimDurationDays')
jcol=claimscluster1.columns.get_loc('ClmDiagnosisCode_1')

##2D Plot####################
#plt.xlabel('Age')
#plt.ylabel('Race')

facet = sns.lmplot(data=claimscluster1, x=claimscluster1.columns[icol], 
                   y=claimscluster1.columns[jcol], hue='Cluster', 
                   fit_reg=False, legend=False)

#add a legend
leg = facet.ax.legend(bbox_to_anchor=[1, 0.75],
                         title="Cluster", fancybox=True)
#change colors of labels
for i, text in enumerate(leg.get_texts()):
    plt.setp(text, color = customPalette[i])


#plt.scatter(clusterdata.iloc[cluster==0, icol], clusterdata.iloc[cluster==0, jcol], s=100, c='b')
#plt.scatter(clusterdata.iloc[cluster==1, icol], clusterdata.iloc[cluster==1, jcol], s=100, c='g')
#plt.scatter(clusterdata.iloc[cluster==2, icol], clusterdata.iloc[cluster==2, jcol], s=100, c='c')
#plt.scatter(clusterdata.iloc[cluster==3, icol], clusterdata.iloc[cluster==3, jcol], s=100, c='r')
#plt.scatter(clusterdata.iloc[cluster==4, icol], clusterdata.iloc[cluster==4, jcol], s=100, c='y')

#plot data with seaborn (don't add a legend yet)


### Displaying cluster#####
display(clusterdata.sort_values(by='Cluster'))
clustersort=clusterdata.groupby(by='Cluster').mean()
clustersort.sort_values("State")

clustersort.to_csv('ClusterOutput.csv')

## #d Plotting########################
color=['autumn','gist_earth','spring','winter','Greens','YlOrRd','turbo','rocket','rainbow']
       

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in np.unique(cluster):
    ax.scatter(claimscluster1.iloc[cluster==i,icol],
               claimscluster1.iloc[cluster==i,jcol],
               claimscluster1.iloc[cluster==i,zcol], marker="o", 
               c=clusterdata.loc[cluster==i,'Cluster'], 
           s=40, cmap=color[i],label=i )

ax.set_xlabel('State')
ax.set_ylabel('ClaimDurationDays')
ax.set_zlabel('PotentialFraud')
ax.legend()
plt.show()



####################################################################################
####################################################################################



## Identifying and dropping outliers#####

outliervariables=claimsdf[['Age','IPAnnualReimbursementAmt','IPAnnualDeductibleAmt',
                  'OPAnnualReimbursementAmt','OPAnnualDeductibleAmt','ClaimDurationDays',
                  'InscClaimAmtReimbursed','DeductibleAmtPaid']]

color = dict(boxes='DarkGreen', whiskers='DarkOrange',
             medians='DarkBlue', caps='Gray')
claimsplotbox = claimsdf[['Age','IPAnnualReimbursementAmt','IPAnnualDeductibleAmt',
                  'OPAnnualReimbursementAmt','OPAnnualDeductibleAmt','ClaimDurationDays',
                  'InscClaimAmtReimbursed','DeductibleAmtPaid']]

claimsplotbox.boxplot()

############### calculate IQR range##############
# identify outliers in Age
Q1 =claimsplotbox.quantile(0.25)
Q3 =claimsplotbox.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

boston_df_out = claimsdf[~((claimsplotbox < (Q1 - 1.5 * IQR)) |
                                (claimsplotbox > (Q3 + 1.5 * IQR))).any(axis=1)]


###########################################################
## Skewness remove ########################################

skewvariables1=claimsdf[['Age','IPAnnualReimbursementAmt','IPAnnualDeductibleAmt',
                  'OPAnnualReimbursementAmt']]
skewvariables2=claimsdf[['OPAnnualDeductibleAmt','ClaimDurationDays',
                  'InscClaimAmtReimbursed','DeductibleAmtPaid']]
#fig, ax = plt.subplots(1, 2) 
  
#sns.distplot(skewvariables['Age'], hist = False, kde = True, 
#            kde_kws = {'shade': True, 'linewidth': 2},  
#            label = "Normal", color ="green", ax = ax[0])

for i, column in enumerate(skewvariables2.columns, 1):
    plt.subplot(2,2,i)
    sns.distplot(skewvariables2[column],hist = False, kde = True,
                  kde_kws = {'shade': True, 'linewidth': 2},  
            label = column, color ="green")


#########################


columns=claimsdf.isna().sum()

### Calculating summary value of each columns

claimsdf_valuesummary= claimsdf[['AttendingPhysician',	'OperatingPhysician',	'OtherPhysician',	'DiagnosisGroupCode',	
'ClmDiagnosisCode_1',	'ClmDiagnosisCode_2',	'ClmDiagnosisCode_3',	'ClmDiagnosisCode_4',
	'ClmDiagnosisCode_5',	'ClmDiagnosisCode_6',	'ClmDiagnosisCode_7',	'ClmDiagnosisCode_8',	
    'ClmDiagnosisCode_9',	'ClmDiagnosisCode_10',	'ClmProcedureCode_1',	'ClmProcedureCode_2',
	'ClmProcedureCode_3',	'ClmProcedureCode_4',	'ClmProcedureCode_5',	'ClmProcedureCode_6']]	

for column in claimsdf_valuesummary:
    writetofile=claimsdf_valuesummary[column].value_counts()
    writetofile.to_csv(column+'.csv')
    
    
    
a='ClmDiagnosisCode_1'
claimsdf_valuesummary[a].value_counts()


claimsdf.ClmDiagnosisCode_1.value_counts()
claimsdf.ClmDiagnosisCode_1.value_counts()

columns.to_csv('columnsselction.csv')
##########End Calculating summary value of each #############


###############################################
######### Chi Square testing Matrix ###########

## Testing association bewteen indepnedent variables######

claimschisq=claimsdf[['Gender','Race','RenalDiseaseIndicator','State',
              'ChronicCond_Alzheimer','ChronicCond_Heartfailure',
                 'ChronicCond_KidneyDisease','ChronicCond_Cancer','ChronicCond_ObstrPulmonary',
                 'ChronicCond_Depression','ChronicCond_Diabetes','ChronicCond_IschemicHeart',
                 'ChronicCond_Osteoporasis','ChronicCond_rheumatoidarthritis','ChronicCond_stroke'
                 ,'AttendingPhysician','ClmDiagnosisCode_1','PotentialFraud']]


claimschisq.dtypes
claimschisq.isna().sum()
claimschisq=claimschisq.dropna()
claimschisq.isna().sum().sum()


column_names=['Gender','Race','RenalDiseaseIndicator','State',
              'ChronicCond_Alzheimer','ChronicCond_Heartfailure',
                 'ChronicCond_KidneyDisease','ChronicCond_Cancer','ChronicCond_ObstrPulmonary',
                 'ChronicCond_Depression','ChronicCond_Diabetes','ChronicCond_IschemicHeart',
                 'ChronicCond_Osteoporasis','ChronicCond_rheumatoidarthritis','ChronicCond_stroke',
                'AttendingPhysician','ClmDiagnosisCode_1','PotentialFraud']



claimschisq.Gender.value_counts()

chisqmatrix=pd.DataFrame(claimschisq,columns=column_names,index=column_names)

#mycrosstab=pd.crosstab(claimschisq['Gender'],claimschisq['Race'])

### Attending physican column has many values with count less than 4
## Dropping values where count is less than 4 

ap_group=claimschisq.AttendingPhysician.value_counts()
ap_group=ap_group[ap_group>5]
ap_group.index

claimschisq=claimschisq[claimschisq.AttendingPhysician.isin(ap_group.index)]

cc_group=claimschisq.ClmDiagnosisCode_1.value_counts()
cc_group=ap_group[cc_group>5]
cc_group.index

claimschisq=claimschisq[claimschisq.ClmDiagnosisCode_1.isin(cc_group.index)]


outercnt=0
innercnt=0
for icol in column_names:
    
    for jcol in column_names:
        
       mycrosstab=pd.crosstab(claimschisq[icol],claimschisq[jcol])
       #print (mycrosstab)
       stat,p,dof,expected=stats.chi2_contingency(mycrosstab)
       chisqmatrix.iloc[outercnt,innercnt]=round(p,3)
       cntexpected=sum(expected<5)
       perexpected=((expected.size-cntexpected)/expected.size)*100
       #print (perexpected)
       #print (icol)
       #print (jcol)
       if perexpected<20:
            chisqmatrix.iloc[outercnt,innercnt]=2
           
       if icol==jcol:
           chisqmatrix.iloc[outercnt,innercnt]=0
       #print (expected) 
       innercnt=innercnt+1
    #print (outercnt) 
    outercnt=outercnt+1
    innercnt=0
    


sns.heatmap(chisqmatrix.astype(np.float64), annot=True,linewidths=0.1,xticklabels=True,
            yticklabels=True,cmap='coolwarm')

### End Chi Square testing#########################

####################################################################
###### Colinearity and Multicolinearity testing#####################
##### Tested on final list of selected NUMERICAL variables####################

## Pick continous columns

claimcolinearity=claimsdf[['DeductibleAmtPaid','InscClaimAmtReimbursed','Age',
                           'OPAnnualDeductibleAmt']]

sns.heatmap(claimcolinearity.corr(), annot=True,linewidths=0.1,xticklabels=True,
            yticklabels=True,cmap='coolwarm')



claimsvif = pd.DataFrame()
claimsvif["feature"] = claimcolinearity.columns
 
# calculating VIF for each feature
claimsvif["VIF"] = [variance_inflation_factor(claimcolinearity.values, i) 
                    for i in range(len(claimcolinearity.columns))]
print(claimsvif)
####################################################################

##########################################################
######Neural Networks#####################################
##########################################################
# Neural networks
# rescale data to [0, 1]
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)



NN = MLPClassifier(solver = 'sgd', alpha = 0.0001,
                   activation='relu', 
                   hidden_layer_sizes = (6,3), random_state = 532)

NN.fit(X_train_scaled, y_train)

NN_pred = NN.predict(X_test_scaled)
metrics.accuracy_score(NN_pred, y_test)
print(metrics.classification_report(y_test, NN_pred))



########## Cross Validation########################

scores = cross_validate(NN, X_scaled, y, scoring ='accuracy', cv=10,
                        return_train_score =True,verbose=1, return_estimator=True,n_jobs=-1)

pd.DataFrame(scores).mean()
# CM and ROC
NN.fit(X_train_scaled, y_train)
NN_pred = NN.predict(X_test_scaled)
metrics.accuracy_score(NN_pred, y_test)
print(metrics.classification_report(y_test, NN_pred))

NN.predict_proba(X_test_scaled)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

######## ROC Curve ###################################

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# predict probabilities
lr_probs = NN.predict_proba(X_test_scaled)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Neural Newtork AUC: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Neural Network')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


########### Optimizer############################
          
print(classification_report(y_true, y_pred))


#@@@@@@

parameter_space = {
    'hidden_layer_sizes': [(30,10),(40,20),(50.30)],
    'activation': ['tanh', 'relu','logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.0005,0.05],
    'learning_rate': ['constant','adaptive'],
}

NN_GCV = MLPClassifier(max_iter =10000, random_state = 210)

#tuned_parameters = {'hidden_layer_sizes':[(10,20,30,40),(20)],
#                    'activation': ['logistic', 'tanh','relu']}

nn_optimizer = GridSearchCV(NN_GCV, parameter_space, 
                            scoring = "accuracy",
                            cv = 10,
                            return_train_score=True, 
                            verbose = 1,n_jobs=-1)

nn_optimizer.fit(X_train_scaled, y)

nn_optimizer.best_params_
nn_optimizer.best_estimator_
nn_optimizer.best_score_

results = pd.DataFrame(nn_optimizer.cv_results_)[['param_activation',
                      'param_hidden_layer_sizes', 'mean_test_score', 
                      'std_test_score', 'rank_test_score']].round(2)

nn_best_model = nn_optimizer.best_estimator_
nn_best_model.fit(X_train_scaled, y_train)

NN_pred_train = nn_best_model.predict(X_train_scaled)
metrics.accuracy_score(NN_pred_train, y_train)
print(metrics.classification_report(y_train, NN_pred_train))

NN_pred_test = nn_best_model.predict(X_test_scaled)
metrics.accuracy_score(NN_pred_test, y_test)
print(metrics.classification_report(y_test, NN_pred_test))
















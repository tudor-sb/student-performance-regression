import pandas as pd
import sklearn.linear_model

df = pd.read_csv("D:\\PythonProjects\\StudentPerformanceFactors.csv")

#Verific pt valori lipsa
print(df.shape)
print(df.info())
print(df.isnull().sum())

#Elmin valorile lipsa 6607->6378 inregistrari
df=df.dropna()
#print(df.isnull().sum())
pd.set_option('display.max_columns', None)
#print(df.describe())

#Inlatur Exam_score=101 outlier
df=df[df["Exam_Score"]<=100]
#print(df.describe(include='object'))

# Studiez distributia exam_score
import matplotlib.pyplot as plt
df.hist("Exam_Score")
plt.xlabel('Exam_Score')
plt.ylabel('Frequency')
plt.title('Distributia Exam_Score')
plt.show()
print(df['Exam_Score'].skew())
print(df['Exam_Score'].kurt())

# Matrice de corelatie initiala

import seaborn as seaborn
seaborn.heatmap(df.select_dtypes(include='number').corr(method='pearson'),
                annot=True,cmap='coolwarm')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Inlocuiesc variabile ordinale cu unele numerice

ordinal_map={'Low': 0,'Medium':1,'High':2,
'Near':0,'Moderate':1,'Far':2,
'High School':0,'College':1,'Postgraduate':2,
'Positive':1,'Negative':-1,'Neutral':0,
'Yes':1,'No':0}

columns = ['Parental_Involvement','Motivation_Level','Peer_Influence',
           'Family_Income', 'Teacher_Quality','Access_to_Resources',
           'Parental_Education_Level', 'Distance_from_Home',
           'Internet_Access', 'Learning_Disabilities', 'Extracurricular_Activities']
for col in columns:
    df[col]=df[col].map(ordinal_map)
print(df.head())
#Matrice de corelatie +var ordinale
plt.figure(figsize=(16, 10))
seaborn.heatmap(df.select_dtypes(include='number').corr(method='pearson'),
                 annot=True,cmap='coolwarm')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#One-hot Encoding
df=pd.get_dummies(df,columns=['Gender','School_Type'])
#print(df.head())

# Linear model
y=df['Exam_Score']
X=df.drop(columns=['Exam_Score'])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,
                                               random_state=16)
from sklearn.linear_model import LinearRegression
model=LinearRegression().fit(X_train,y_train)
y_predicted=model.predict(X_test)

import sklearn.metrics
#Indicatori de performanta pentru regresie
Scor_R=round(sklearn.metrics.r2_score(y_test,y_predicted),3)
RMSE=round(sklearn.metrics.root_mean_squared_error(y_test,y_predicted),3)
MAE=round(sklearn.metrics.mean_absolute_error(y_test,y_predicted),3)
print("Valoarea R Square pentru modelul simplu este:",Scor_R)
print("Valoarea RMSE pentru modelul simplu este:",RMSE)
print("Valoarea MAE pentru modelul simplu este:",MAE)

y_reziduuri=y_test-y_predicted
#Grafic reziduuri:
seaborn.residplot(x=y_predicted,y=y_test)
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Graficul Reziduurilor")
plt.show()

#Feature importance
coeficienti=pd.DataFrame(model.coef_,X.columns,columns=["Importanta"])
print(coeficienti.sort_values(by="Importanta",ascending=False))

#Random Forest -> o suma de decision trees -> estimator puternic ensemble learning
from sklearn.ensemble import RandomForestRegressor

regr_model=RandomForestRegressor(n_estimators=100,max_depth=7,random_state=42)
regr_model.fit(X_train,y_train)
y_predicted_regr=regr_model.predict(X_test)

#Metrics RFR
Scor_R_regr=round(sklearn.metrics.r2_score(y_test,y_predicted_regr),3)
RMSE_regr=round(sklearn.metrics.root_mean_squared_error(y_test,y_predicted_regr),3)
MAE_regr=round(sklearn.metrics.mean_absolute_error(y_test,y_predicted_regr),3)
print("Valoarea R Square pentru modelul RFR este:",Scor_R_regr)
print("Valoarea RMSE pentru modelul RFR este:",RMSE_regr)
print("Valoarea MAE pentru modelul RDR este:",MAE_regr)
print(regr_model.score(X_train,y_train)) # Overfitting si cu modificari la arbore modelul nu e suficient de eficient, raman la Linear Regression

#Coeficienti RFR
coeficienti_rfr=pd.DataFrame(regr_model.feature_importances_,X.columns,columns=["Importanta"])
print(coeficienti_rfr.sort_values(by="Importanta",ascending=False))

metrics = ['R²', 'RMSE', 'MAE']
lr_scores = [Scor_R, RMSE, MAE]
rfr_scores = [Scor_R_regr, RMSE_regr, MAE_regr]

#Grafic comparatie metrics LR si RFR
x = range(len(metrics))
plt.figure(figsize=(10, 6))
plt.bar([i - 0.2 for i in x], lr_scores, width=0.4, label='Linear Regression', color='blue')
plt.bar([i + 0.2 for i in x], rfr_scores, width=0.4, label='Random Forest', color='green')
plt.xticks(x, metrics)
plt.title('Comparatie modele')
plt.legend()
plt.show()

import statsmodels.api as sm
X_statsmodels = X.drop(columns=['Gender_Female', 'School_Type_Private']).astype(float)
X_cu_const = sm.add_constant(X_statsmodels)
model_detaliat = sm.OLS(y, X_cu_const).fit()
print(model_detaliat.summary())
# skewness ridicat in reziduuri cauzat de distributia concentrata a Exam_Score
# (maj intre 63-69) cu putine valori extreme. Modelul ramane valid totusi

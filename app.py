import pandas as pd
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import dash_bootstrap_components as dbc
from dash import html, dcc
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
import dash_html_components as html
import dash_bootstrap_components as dbc
from explainerdashboard.custom import *
from explainerdashboard import ExplainerDashboard

df = pd.read_excel("/Users/manekasingh/Desktop/test/dashboard/StudentsMarks.xlsx")

print(df)

print(df.columns)

df=df.sample(frac=1).reset_index(drop=True)

X = df.iloc[:, 0:7]
y = df[["Class"]]

y.head(2)

y["Class"] = y["Class"].astype("category")
y["Students_Performance"] = y["Class"].cat.codes

y = y[["Students_Performance"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=55, stratify=y)


model = XGBClassifier().fit(X_train,y_train)

explainer = ClassifierExplainer(model, X_test, y_test)



db = ExplainerDashboard(explainer, title="Teacher's Dashboard", name="dash", description="dashboard", bootstrap=dbc.themes.FLATLY, 
                    whatif=False, header_hide_selector=True, header_hide_download=True, hide_poweredby=True, shap_interaction=True).run()


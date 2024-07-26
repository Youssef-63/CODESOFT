import pandas as pd
from sklearn.datasets import load_iris
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error,confusion_matrix
from sklearn.model_selection import train_test_split

st.write("""
        # Iris Flower Classification
""")
st.sidebar.header("Input Features")

data = load_iris()
X = data.data
y = data.target

st.subheader('Training data')
st.write(pd.DataFrame(X, columns=data.feature_names).head())
st.write('Features shape : ',X.shape)
st.write('Flower types : ')
st.write(pd.DataFrame({'flower':data.target_names}))

def predict_flower():
    sepal_length = st.sidebar.slider('Sepal length',4.3,7.9,5.8)
    sepal_width = st.sidebar.slider('Sepal width',2.0,4.4,3.0)
    petal_length = st.sidebar.slider('Petal length',1.0,6.9,3.7)
    petal_width = st.sidebar.slider('Petal width',0.1,2.5,1.2)
    df = {
        'sepal_length':[sepal_length],
        'sepal_width':[sepal_width],
        'petal_length':[petal_length],
        'petal_width':[petal_width]
    }
    df = pd.DataFrame(df)
    return df


X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)
ytr_pred = rf.predict(X_train)
yt_pred = rf.predict(X_test)

train_acc = accuracy_score(y_train, ytr_pred)
train_mse = mean_squared_error(y_train,ytr_pred)
train_coff = confusion_matrix(y_train,ytr_pred)

test_acc = accuracy_score(y_test, yt_pred)
test_mse = mean_squared_error(y_test,yt_pred)
test_coff = confusion_matrix(y_test,yt_pred)

metrics = {
    'Train accuracy':train_acc,
    'Train MSE':train_mse,
    'Test accuracy':test_acc,
    'Test MSE':test_mse
}

st.subheader('Model Training Metrics')
st.write(pd.DataFrame(metrics, index=[0]))
st.write('Train Confusion Matrix',train_coff,'Test Confusion Matrix',test_coff)


df = predict_flower()
st.subheader('Input Features to Predict Flower')
st.write(df)

pred = rf.predict(df)
pred_prb = rf.predict_proba(df)
pred_prb = pd.DataFrame(pred_prb, columns=data.target_names)

st.subheader('Prediction Probability')
st.write(pred_prb)

st.subheader('Prediction')
st.write(data.target_names[pred])

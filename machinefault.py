import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

#df = pd.read_csv(r"G:\Users\dabra\downloads\predictive_maintenance.csv")
df=pd.read_csv("/content/predictive_maintenance (1).csv")
del df['UDI']
del df['Type']
del df['Product ID']
label = df['Target'].to_numpy()
features = ['Air temperature [K]', 'Process temperature [K]','Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
features = df[features]

scaler = MinMaxScaler()
Xtransformed = scaler.fit_transform(features)
Xtrain = Xtransformed[:6000,:]
trainLabel = label[:6000,]
model = KNeighborsClassifier(n_neighbors=3)
model=model.fit(Xtrain,trainLabel)
def main():
    st.title('Machine Fault Detection with kNN')
    Airtemperature= st.slider('Air Temperature', 295, 310, 300)
    Processtemperature= st.slider('Process Temperature', 306, 314, 310)
    Rotationalspeed= st.slider('Rotational speed', 1168, 2886, 1452)
    Torque= st.slider('Torque', 4, 77, 40)
    Toolwear = st.slider('Tool Wear', 0, 253, 0)

    # Create a feature array with the user's input
    features1 = np.array([[Airtemperature, Processtemperature, Rotationalspeed, Torque, Toolwear]])

    # Make predictions using the kNN model
    prediction = model.predict(features1)
    if prediction == 1:
        predicted_label = 'Faulty'
    else:
        predicted_label = 'Not Faulty'

        # Display the prediction
    st.write(f'The Machine is : {predicted_label}')


if __name__ == '__main__':
    main()

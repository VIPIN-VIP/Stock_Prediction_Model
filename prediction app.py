import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start="2012-01-01"
end = "2021-12-31"
st.title('Stock prediction')

user_input =st.text_input('Enter Stock Ticker', 'AAPL')
df =data.DataReader(user_input, 'yahoo', start, end)

#Describing data
clo1=df.reset_index()['Close']

st.subheader('Data from 2012-2021')
st.write(df.describe())

#Visualization

st.subheader('Closing Price vs Time chart ')
fig =plt.figure(figsize=(12,6))
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(clo1)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart witth 100 days moving avg')
rollavg100=clo1.rolling(100).mean()
fig =plt.figure(figsize=(12,6))
plt.plot(rollavg100,'r',label='rollavg100')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.plot(clo1)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart witth 100 & 200 days moving avg')
rollavg100=clo1.rolling(100).mean()
rollavg200=clo1.rolling(200).mean()
fig =plt.figure(figsize=(12,6))
plt.plot(rollavg100,'r',label='rollavg100')
plt.plot(rollavg200,'g',label='rollavg200')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.plot(clo1)
st.pyplot(fig)

#Splitting data into traning and testing

data_training=pd.DataFrame(clo1[0:int(len(clo1)*0.7)])
data_testing=pd.DataFrame(clo1[int(len(clo1)*0.7):int(len(clo1))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#Model load

model = load_model('Stock_Market_prediction.h5')

#testing part

last_100_days=data_training.tail(100)
finaldata= last_100_days.append(data_testing,ignore_index=True)
clo2= scaler.fit_transform(finaldata)
lis2=[]
pre2=[]
for i in range(100,clo2.shape[0]):
    lis2.append(clo2[i-100: i])
    pre2.append(clo2[i, 0])
lis2,pre2=np.array(lis2),np.array(pre2)

pre_prediction=model.predict(lis2)


scaler= scaler.scale_
scale_factor=1/scaler[0]
pre_prediction=pre_prediction*scale_factor
pre2=pre2*scale_factor

# Final graph
st.subheader('Prediction vs Original')
figu=plt.figure(figsize=(12,6))
plt.plot(pre2,'r',label='Original Price')
plt.plot(pre_prediction,'g',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(figu)




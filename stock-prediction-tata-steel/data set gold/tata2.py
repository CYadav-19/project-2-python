import pandas as pd
import matplotlib.pyplot as plt

gstock_data=pd.read_csv('tata-steel.csv')

gstock_data.head()


#We will use opening and closing values for our experimentation of time series with LSTM
gstock_data = gstock_data [['Date','Open','Close']] 
gstock_data['Date'] = pd.to_datetime(gstock_data['Date'].apply(lambda x: x.split()[0]))
gstock_data .set_index('Date',drop=True,inplace=True) 
gstock_data .head()

#Now we can be using matplotlib to visualize the available data 
#to  see how our price values in data are being displayed. 
#The green colour was used to visualize the open variable for the price-date graph,
# for the closing variable we used red colour.

fig, ax = plt.subplots(1, 2, figsize=(20, 7))

ax[0].plot(gstock_data ['Open'],label='Open',color='green')
ax[0].set_xlabel('Date',size=15)
ax[0].set_ylabel('Price',size=15)
ax[0].legend()

ax[1].plot(gstock_data ['Close'],label='Close',color='red')
ax[1].set_xlabel('Date',size=15)
ax[1].set_ylabel('Price',size=15)
ax[1].legend()

fig.show()


# We must pre-process this data before applying stock price using LSTM
from sklearn.preprocessing import MinMaxScaler

Ms = MinMaxScaler()

gstock_data [gstock_data .columns] = Ms.fit_transform(gstock_data )
training_size = round(len(gstock_data ) * 0.80)
train_data = gstock_data [:training_size]
test_data  = gstock_data [training_size:]


#A function is created so that we can create the sequence for training and testing.


def create_sequence(<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..dataset'}, '*')">dataset):
    <a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..sequences'}, '*')">sequences = []
    <a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..labels'}, '*')">labels = []
    <a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..start_idx'}, '*')">start_idx = 0
    for <a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..stop_idx'}, '*')">stop_idx in range(50,len(<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..dataset'}, '*')">dataset)): 
        <a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..sequences'}, '*')">sequences.append(<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..dataset'}, '*')">dataset.iloc[<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..start_idx'}, '*')">start_idx:<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..stop_idx'}, '*')">stop_idx])
        <a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..labels'}, '*')">labels.append(<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..dataset'}, '*')">dataset.iloc[<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..stop_idx'}, '*')">stop_idx])
        <a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..start_idx'}, '*')">start_idx += 1
    return (np.<a onclick="parent.postMessage({'referent':'.numpy.array'}, '*')">array(<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..sequences'}, '*')">sequences),np.<a onclick="parent.postMessage({'referent':'.numpy.array'}, '*')">array(<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..labels'}, '*')">labels))

train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)


































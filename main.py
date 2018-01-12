# # Weather Prediction Example
# This template shows how to fetch weather data for a particular zip code from the Metis Machine data engine, and then use those data to train a dummy [recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) that will predict some aspect of the weather from the historical time series at that location.
# Obviously, weather modeling is not actually this easy, but the following code shows how to:
# * access data via the Metis Machine data engine
# * transform those data using a deep learning model
# * persist the resulting transformation so that it can be accessed via an outward-facing API

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from datetime import datetime
import pandas as pd


# ## Access weather data using the Metis Machine SDK
# Any data intake or model project on the platform begins by initializing the Skafos SDK.
# This allows your task to access the resources of the platform, as well as ensures proper health monitoring.
#
# *Unresponsive tasks will eventually be purged.*

from skafossdk import *
print('Initializing the SDK connection')
skafos = Skafos()


res = skafos.engine.create_view(
    "weather_noaa", {"keyspace": "weather",
                      "table": "weather_noaa"}, DataSourceType.Cassandra).result()
print("created a view of NOAA historial weather data")


print("pulling historical weather from a single zip code")
weather_json = skafos.engine.query("SELECT * from weather_noaa WHERE zipcode = 23250").result()


# validate a single record
weather_json['data'][0]


# convert retrieved records to a dataframe
import pandas as pd
weather = pd.DataFrame(weather_json['data'])
weather['date']  = pd.to_datetime(weather['date'])


# validate this is what we expect, missing values may throw off pandas types
weather.info()


# some numerical values do not show as such, clean missing records
weather['precip_total'] = weather['precip_total'].replace('NaN', None, regex=False).fillna(0)
weather['pressure_avg'] = weather['pressure_avg'].replace('NaN', None, regex=False).fillna(0)
weather['wind_speed_peak'] = weather['wind_speed_peak'].replace('NaN', None, regex=False).fillna(0)


# verify that we now have all the float columns we expected
weather.info()


# # Prep inputs for modeling
# We want to use a recurrent time-series model, so our data need to be in ascending order by date.

day_zero = weather['date'].min()


weather.set_index((weather['date'] - day_zero).apply(lambda d: d.days), inplace=True)
weather.sort_index(inplace=True)


# ## Feature Engineering
# These are not necessarilly excellent features, but simply illustrate a common step in the predictive process.
#
# * length of day
# * average temperature
# * change in average temperature
# * change in barometric pressure
# * precipitation
# * wind speed peak

weather['precip_total'].fillna(0, inplace=True)


weather['day_length'] = weather.apply(lambda r: int(r.sunset) - int(r.sunrise), axis=1)


weather['tavg'] = (weather.tmax + weather.tmin) / 2


weather['pressure_change'] = weather['pressure_avg'].pct_change()


weather['temp_change'] = weather['tavg'].pct_change()


weather_features = weather[
    ['day_length', 'tavg', 'tmin', 'tmax', 'temp_change', 'pressure_change', 'precip_total', 'wind_speed_peak']].dropna()


# validate inputs to the RNN
weather_features.iloc[:6]


# ## Normalize inputs for deep learning
# Most neural networks expect inputs from -1 to 1

# fit two standard deviations between -1 and 1
weather_norm = weather_features.apply(lambda c: 0.5 * (c - c.mean()) / c.std())


weather_x = weather_norm.drop('tavg', axis=1)
# shift so that we're trying to predict tomorrow
weather_y = weather_norm['tavg'].shift(-1)


# predict on the last two months
predict_day = weather_x.index[-60]


# # Recurrent Neural Network Model
# [PyTorch](http://pytorch.org) is a wonderful framnework for deep learning since it handles backpropgation automatically.

x_train = torch.autograd.Variable(
    torch.from_numpy(weather_x.loc[:predict_day - 1].as_matrix()).float(), requires_grad=False)
x_test = torch.autograd.Variable(
    torch.from_numpy(weather_x.loc[predict_day:].as_matrix()).float(), requires_grad=False)
batch_size = x_train.size()[0]
input_size = len(weather_x.columns)


y_train = torch.autograd.Variable(
    torch.from_numpy(weather_y.loc[:predict_day - 1].as_matrix()).float(), requires_grad=False)
y_test = torch.autograd.Variable(
    torch.from_numpy(weather_y.loc[predict_day:].as_matrix()).float(), requires_grad=False)


class WeatherNet(torch.nn.Module):
    hidden_layers = 2
    hidden_size = 6

    def __init__(self):
        super(WeatherNet, self).__init__()
        # use a small hidden layer since we have such narrow inputs
        self.rnn1 = nn.GRU(input_size=input_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.hidden_layers)
        self.dense1 = nn.Linear(self.hidden_size, 1)

    def forward(self, x, hidden):
        x_batch = x.view(len(x), 1, -1)
        x_r, hidden = self.rnn1(x_batch, hidden)
        x_o = self.dense1(x_r)
        return x_o, hidden

    def init_hidden(self, batch_size):
        return Variable(torch.randn(self.hidden_layers, 1, self.hidden_size))


# ### Train the RNN
# Initialize the hidden layer during training, but keep it for later prediction.

torch.manual_seed(0)
model = WeatherNet()
print(model)
criterion = nn.MSELoss(size_average=True)
optimizer = torch.optim.Adadelta(model.parameters())


hidden = model.init_hidden(batch_size)

for i in range(120):
    def closure():
        model.zero_grad()
        hidden = model.init_hidden(batch_size)
        out, hidden = model(x_train, hidden)
        loss = criterion(out, y_train)
        if i % 10 == 0:
            print('{:%H:%M:%S} epoch {} loss: {}'.format(datetime.now(), i, loss.data.numpy()[0]), flush=True)
        loss.backward()
        return loss
    optimizer.step(closure)


# # Predict
# Keep the current hidden state of the model and run it forward without updating parameters

y_pred, new_hidden = model(x_test, hidden)


predictions = pd.DataFrame(y_pred.view(len(y_pred), -1).data.numpy(), columns=['tavg_norm'])
predictions['series'] = 'predicted'


actuals = pd.DataFrame(y_test.data.numpy(), columns=['tavg_norm'])
actuals['series'] = 'actual'


# join for plotting purposes
eval_data = pd.concat([predictions, actuals])
eval_data['day'] = eval_data.index


# ### UnNormalize Predictions for Display
# This was how we normalized the inputs to the RNN, we will just undo that transformation for plotting purposes.
# ``` python
# weather_norm = weather_features.apply(lambda c: 0.5 * (c - c.mean()) / c.std())
# ```

eval_data['tavg'] = 2. * eval_data['tavg_norm'] * weather_features['tavg'].std() + weather_features['tavg'].mean()

# # Persist Predictions

# define the schema for this dataset
schema = {
    "table_name": "rnn_weather_predictions",
    "options": {
        "primary_key": ["day", "series"],
        "order_by": ["series asc"]
    },
    "columns": {
        "day": "int",
        "tavg": "float",
        "series": "text"
    }
}


data_out = eval_data.dropna().drop('tavg_norm', axis=1).to_dict(orient='records')


skafos.engine.save(schema, data_out).result()


# ## Accessing persisted data
# Ingested data is available from the Metis Machine API using your credentials as described in the [API docs](https://docs.metismachine.io/docs/api-accessing-your-results)

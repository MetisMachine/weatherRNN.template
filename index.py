from skafossdk import *
print('initialize the SDK connection')
skafos = Skafos()

res = skafos.engine.create_view(
    "weather_noaa", {"keyspace": "weather",
                      "table": "weather_noaa"}, DataSourceType.Cassandra).result()
print("created a view of NOAA historial weather data")

print("pulling historical weather from a single zip code")
weather_json = skafos.engine.query("SELECT * from weather_noaa WHERE zipcode = 23250").result()

# convert to a dataframe
import pandas as pd
weather = pd.DataFrame(weather_json['data'])
weather['date']  = pd.to_datetime(weather['date'])

# fix python crazy with missing values
weather['precip_total'] = weather['precip_total'].replace('NaN', None, regex=False).fillna(0)
weather['pressure_avg'] = weather['pressure_avg'].replace('NaN', None, regex=False).fillna(0)
weather['wind_speed_peak'] = weather['wind_speed_peak'].replace('NaN', None, regex=False).fillna(0)


# # Prep inputs for modeling
# First we make a numeric time scale, though just having them in order of date is sufficient
day_zero = weather['date'].min()

weather.set_index((weather['date'] - day_zero).apply(lambda d: d.days), inplace=True)
weather.sort_index(inplace=True)


## Create features
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

weather_features.iloc[:6]


## Normalize inputs for deep learning
# Most neural networks expect inputs from -1 to 1

weather_norm = weather_features.apply(lambda c: 0.5 * (c - c.mean()) / c.std())
weather_x = weather_norm.drop('tmax', axis=1)
# shift so that we're trying to predict tomorrow
weather_y = weather_norm['tmax'].shift(-1)

# predict on the last two months
predict_day = weather_x.index[-60]


# # Recurrent Neural Network Model
# Build a PyTorch model to do time series prediction

import torch
import torch.nn as nn
from torch.autograd import Variable
from datetime import datetime
import pandas as pd
import ggplot as gg


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


torch.manual_seed(0)
model = WeatherNet()
print(model)
criterion = nn.MSELoss(size_average=True)
# use LBFGS as optimizer since we can load the whole data to train
#optimizer = optim.LBFGS(seq.parameters())
optimizer = torch.optim.Adadelta(model.parameters())

hidden = model.init_hidden(batch_size)

for i in range(72):
    def closure():
        model.zero_grad()
        hidden = model.init_hidden(batch_size)
        out, hidden = model(x_train, hidden)
        loss = criterion(out, y_train)
        if i % 2 == 0:
            print('{:%H:%M} epoch {} loss: {}'.format(datetime.now(), i, loss.data.numpy()[0]))
        loss.backward()
        return loss
    optimizer.step(closure)


# # Predict
# Keep the current hidden state of the model and run it forward without updating parameters

y_pred, new_hidden = model(x_test, hidden)

predictions = pd.DataFrame(y_pred.view(len(y_pred), -1).data.numpy(), columns=['tmax_norm'])
predictions['actual'] = 'predicted'

actuals = pd.DataFrame(y_test.data.numpy(), columns=['tmax_norm'])
actuals['actual'] = 'actual'

# join for plotting purposes
eval_data = pd.concat([predictions, actuals])
eval_data['day'] = eval_data.index

# # Write data out
# need a schema to write data out
schema = {
    "table_name": "my_weather_predictions",
    "options": {
        "primary_key": ["day", "actual"],
        "order_by": ["actual asc"]
    },
    "columns": {
        "day": "int",
        "tmax_norm": "float",
        "actual": "text"
    }
}


data_out = eval_data.dropna().to_dict(orient='records')
dataresult = skafos.engine.save(schema, data_out).result()


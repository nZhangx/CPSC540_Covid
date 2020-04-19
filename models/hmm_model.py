#540 project
from statsmodels.tsa.ar_model import AR
from random import random
import pandas as pd
import numpy as np
import numpy as np
from hmmlearn import hmm
import seaborn as sns
%matplotlib qt
%matplotlib inline

import matplotlib.pyplot as plt
covid = pd.read_csv('/Users/Ishika/Desktop/csse_world_normalized.txt')
covid

covid.drop(covid.columns[[0]], axis=1, inplace=True)
covid.drop(covid.columns[[0]], axis=1, inplace=True)
total_infec = covid.sum(axis = 0)
total_infec = pd.DataFrame(total_infec)
total_infec = np.array(total_infec)
total_infec.shape
Italy
#average samples by day
model = hmm.GaussianHMM(n_components = 4, covariance_type="full", n_iter = 1000)
model = model.fit(total_infec)
#model
#model.transmat_
num_samples = 84
samples, states = model.sample(num_samples)
model.decode(total_infec)

samples.shape
plt.plot(samples)
plt.plot(total_infec)
plt.show()

mse = (samples - total_infec)**2
mse = mse.sum(axis = 0)
mse


Z = model.predict(total_infec)
Z
states
model.decode(total_infec)
model.score_samples(total_infec)

total_infec
### loop through to average samples

samples = []
samples, states = model.sample(num_samples)
samples = pd.DataFrame(samples)
for i in range(100):
    samples_new, states = model.sample(num_samples)
    samples_new = pd.DataFrame(samples_new)
    samples = pd.concat([samples, samples_new], axis=1)



average_samples = pd.DataFrame(samples.mean(axis=1))

plt.plot(average_samples)
plt.plot(total_infec)
plt.show()
average_samples.shape

mse = (average_samples - total_infec)**2
mse = sqrt(mse.sum(axis = 0))
mse


#####################Country Specific Models##############################

value_list = ['Italy']
Italy = pd.DataFrame(covid[covid['Country/Region'].isin(value_list)])
Italy.drop(Italy.columns[[0]], axis=1, inplace=True)

Italy = Italy.transpose()
Italy.shape
model = hmm.GaussianHMM(n_components = 4, covariance_type="full", n_iter = 1000)
model = model.fit(Italy)
#model
#model.transmat_
num_samples = 84
samples, states = model.sample(num_samples)
model.decode(total_infec)

samples.shape
plt.plot(samples)
plt.plot(Italy)
plt.show()

mse = (samples - total_infec)**2
mse = sqrt(mse.sum(axis = 0))
mse


#averaged
samples = []
samples, states = model.sample(num_samples)
samples = pd.DataFrame(samples)
for i in range(100):
    samples_new, states = model.sample(num_samples)
    samples_new = pd.DataFrame(samples_new)
    samples = pd.concat([samples, samples_new], axis=1)



average_samples = pd.DataFrame(samples.mean(axis=1))
average_samples.columns = ['counts']
plt.plot(average_samples)
plt.plot(Italy)
plt.show()


average_samples
#sns.lineplot(x = average_samples.index.values,y = average_samples['counts'])
average_samples.shape

mse = (average_samples - total_infec)**2
mse = sqrt(mse.sum(axis = 0))
mse


#########try using countries as full samples

value_list = ['Italy']
Italy = pd.DataFrame(covid[covid['Country/Region'].isin(value_list)])
Italy.drop(Italy.columns[[0]], axis=1, inplace=True)

Italy = Italy.transpose()
Italy.shape

value_list = ['US']
US = pd.DataFrame(covid[covid['Country/Region'].isin(value_list)])
US.drop(US.columns[[0]], axis=1, inplace=True)

US = US.transpose()
US.shape

X = np.concatenate([Italy, US])
lengths = [len(Italy), len(US)]
#hmm.GaussianHMM(n_components=3).fit(X, lengths)

model = hmm.GaussianHMM(n_components = 4, covariance_type="full", n_iter = 1000)
model = model.fit(X, lengths)
#model
#model.transmat_
num_samples = 84
samples, states = model.sample(num_samples)
model.decode(total_infec)
samples.shape
samples.shape
plt.plot(samples)
plt.plot(X)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import pandas as pd
from scipy.spatial import distance


x = np.loadtxt(r'C:\Users\vicen\OneDrive\Escritorio\Memoria\ModeloEC\dataset\scandec.csv', delimiter=',')

fourier_library = ps.FourierLibrary(n_frequencies=2, include_cos=True, include_sin=True)
optimizer = ps.STLSQ(threshold=0.0, fit_intercept=True)

model = ps.SINDy(feature_library=fourier_library, optimizer=optimizer ,feature_names=['x' , 'y'])
print(model)
t = np.arange(0, 300, 1)
model.fit(x, t=t)
model.print() 

x_model = model.simulate(x[0, :], t)
distancia = distance.cdist(x[:, :2], x_model[:, :2], 'euclidean')
distancia_media = distancia.mean()
distancia_maxima = np.max(distancia)

print(f'La distancia media entre ambas curvas es {distancia_media:.2f}')
print(f'La distancia media entre ambas curvas es {distancia_maxima}')

fig, ax, = plt.subplots(1, 1 , figsize=(6,6))
ax.plot(x[:, 0], x[:, 1], label='Modelo base')
ax.plot(x_model[:, 0], x_model[:, 1], '--' , label='Modelo PySINDy')
ax.set(xlabel='Tiempo', ylabel='Concentración', title = 'Curva biomasa')
ax.legend
fig.show()


plt.show()



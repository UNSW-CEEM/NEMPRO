import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import pandas as pd

forecast_sensitivities = pd.read_csv('forecast_sensitivities.csv')

x = forecast_sensitivities['nsw-demand']
y = forecast_sensitivities['qld-demand']
z = forecast_sensitivities['nsw-energy']

xi, yi = np.meshgrid(np.linspace(0, 3, 20), np.linspace(0, 3, 20))


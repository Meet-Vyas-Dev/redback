import redback
import pandas as pd
from redback.simulate_transients import SimulateOpticalTransient
import matplotlib.pyplot as plt
import numpy as np

# specify the number of pointings per band 
num_obs = {'lsstg': 10, 'lsstr':10, 'lssti':10, 'lsstz':10, 'lsstu':10}

# specify the cadence in days for each band
average_cadence = {'lsstg': 1.5, 'lsstr': 5.0, 'lssti': 2.5, 'lsstz':1, 'lsstu':1}

# specify any scatter on the cadence, the time of the observation will be 
# taken from a Gaussian with the scatter as sigma
cadence_scatter = {'lsstg': 0.5, 'lsstr':0.5, 'lssti':0.5, 'lsstz':1, 'lsstu':1}

# Specify limiting 5 sigma depth magnitude
limiting_magnitudes = {'lsstg': 25.0, 'lsstr': 24.5, 'lssti': 23.0, 'lsstu':25, 'lsstz':23}

# We now use redback to make a pointings table from the above information
# We set RA and DEC to always be at the location of the transient 
# but we can change this to incorporate the fov/full survey
ra = 1.0 
dec = 1.5
# We also set the start time of the observation/survey strategy 
initMJD = 59581.0
pointings = redback.simulate_transients.make_pointing_table_from_average_cadence(
    ra=ra, dec=dec, num_obs=num_obs, average_cadence=average_cadence,
    cadence_scatter=cadence_scatter, limiting_magnitudes=limiting_magnitudes, 
    initMJD=59581.0)
print(pointings)

model_kwargs = {}
# Any redback model can be referred to as a string. 
# If the user has their own model, they can pass a function here instead. 
# There are over a 100 models implemented in redback, lots of models for kilonovae, GRB afterglows, 
# supernovae, TDEs and other things
model = 'shock_cooling'
# Load the default prior for this model in redback and sample from it to get 1 set of parameters. 
# We can sample from the default prior for this model for a random kilonova. 
parameters = redback.priors.get_priors(model=model).sample()

# We fix a few parameters here to create a nice looking kilonova. 
# You can change any of the parameters here or add additional keyword arguments 
# to change some physical assumptions. Please refer to the documentation for this and units etc
parameters['mej'] = 0.05
parameters['t0_mjd_transient'] = 59582.0
parameters['redshift'] = 0.075
parameters['t0'] = parameters['t0_mjd_transient']
parameters['temperature_floor'] = 3000
parameters['kappa'] = 1
parameters['vej'] = 0.2
parameters['ra'] = 1.0
parameters['dec'] = 1.5
print(parameters)

# Specify some additional settings. 
# A threshold for data points to consider detections based on a SNR. 
snr_threshold = 5.0

# A maximum time to evaluate the transient, 
# this is useful if you do not trust the model past a certain time or do not want to generate detections past this time. 
end_transient_time = 20

# Simulate by passing in the parameters of the model, the model string (or a python function), 
# and the pointings generated above.
kn_sim = SimulateOpticalTransient.simulate_transient(model='shock_cooling',
                                       parameters=parameters, pointings_database=pointings,
                                       survey=None, model_kwargs=model_kwargs,
                                        end_transient_time=20., snr_threshold=5.0)

# We can print the observations that were simulated to see what the data looks like. 
# This will include extra stuff like non-detections etc
print(kn_sim.observations)

import pandas as pd

# Load the CSV file
csv_file_path = 'shock_cooling.csv'  # Ensure this path matches where your CSV is saved
observations_df = pd.read_csv(csv_file_path)

# Save to Excel format
excel_file_path = 'shock_cooling.xlsx'
observations_df.to_excel(excel_file_path, index=False)

# Now read the Excel file using Pandas
observations_df_from_excel = pd.read_excel(excel_file_path)

# Assuming Redback needs a path to a CSV
kn_object = redback.transient.Kilonova.from_simulated_optical_data(name=csv_file_path, data_mode='magnitude')

# Make a dictionary for colors on the plot
band_colors = {'lsstg':'#4daf4a', 'lsstu':'#377eb8', 'lsstr':'#e41a1c', 
               'lsstz':'#a65628', 'lssti':'#ff7f00'}
ax = kn_object.plot_data(show=False, band_colors=band_colors)
# ax.set_ylim(22, 19)

# Make a dictionary for colors on the plot
band_colors = {'lsstg':'#4daf4a', 'lsstu':'#377eb8', 'lsstr':'#e41a1c', 
               'lsstz':'#a65628', 'lssti':'#ff7f00'}
ax = kn_object.plot_data(show=False, band_colors=band_colors)
ax.set_ylim(28, 22)
upper_limits = kn_sim.observations[kn_sim.observations['detected'] != 1.0]
data = kn_sim.observations[kn_sim.observations['detected'] == 1.0]
for band in band_colors.keys():
    up = upper_limits[upper_limits['band'] == band]
    dd = data[data['band'] == band]
    plt.scatter(dd['time (days)'], dd['magnitude'], s=100, marker='.', color=band_colors[band])
    plt.scatter(up['time (days)'], up['limiting_magnitude'], s=100, marker=r'$\downarrow$', color=band_colors[band])

    
# We can also plot the true data 
tt = np.linspace(0.1, 20, 100)
# specify output_format 
parameters['output_format'] = 'magnitude'
for band in band_colors.keys():
    parameters['bands'] = band
    out = redback.transient_models.kilonova_models.one_component_kilonova_model(tt, **parameters)
    plt.plot(tt, out, color=band_colors[band], alpha=0.3)

plt.xlim(0.1, 10)
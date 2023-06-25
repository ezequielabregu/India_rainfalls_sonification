import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import hann

# Load the data from the CSV file
data = pd.read_csv('annual_data.csv')

def sonify_data(data, duration=0.03, fade_duration=0.01, sampling_freq=44100, output_folder='sonified_audio'):
    # Extract the unique regions from the 'SUBDIVISION' column
    regions = data['SUBDIVISION'].unique()

    # Create a time array
    t = np.linspace(0, duration, int(duration * sampling_freq), endpoint=False)

    # Create a fade array
    fade_samples = int(fade_duration * sampling_freq)
    fade_window = hann(fade_samples * 2)[:fade_samples]

    # Find the minimum and maximum frequency values across all regions
    min_frequency = np.min(data['ANNUAL'])
    max_frequency = np.max(data['ANNUAL'])

    # Sonify data for each region
    for region in regions:
        # Filter the data for the current region
        region_data = data[data['SUBDIVISION'] == region]

        # Extract the 'YEAR' and 'ANNUAL' columns for the current region
        years = region_data['YEAR']
        annual_data = region_data['ANNUAL']

        # Create an empty array to store the sonified signal
        sonified_signal = np.array([])

        # Create empty lists to store the x and y values for the scatter plot
        scatter_x = []
        scatter_y = []

        # Sonify each data point using a sine wave and apply smooth start and end
        for year, data_point in zip(years, annual_data):
            frequency = data_point  # Use the data point directly as frequency

            # Apply smooth start and end using fade-in and fade-out
            fade_in = fade_window[:fade_samples]
            fade_out = fade_window[-fade_samples:]

            tone = np.sin(2 * np.pi * frequency * t)
            tone[:fade_samples] *= fade_in
            tone[-fade_samples:] *= fade_out

            sonified_signal = np.concatenate([sonified_signal, tone])

            # Repeat the year value for each sample in the data point
            scatter_x.extend([year] * len(t))
            scatter_y.extend([frequency] * len(t))

        # Normalize the sonified signal
        sonified_signal /= np.max(np.abs(sonified_signal))

        # Save the sonified signal as a WAV audio file
        output_file = f"{output_folder}/{region.replace(' ', '_')}.wav"
        sf.write(output_file, sonified_signal, sampling_freq)

        # Create a scatter plot of frequencies over time
        plt.scatter(scatter_x, scatter_y, marker='o', s=10)
        plt.xlabel('Year')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'Sonification Plot - {region}')
        plt.ylim(min_frequency, max_frequency)  # Set the Y-axis limits based on the frequency range
        plt.savefig(f"{output_folder}/{region.replace(' ', '_')}_plot.png")
        plt.close()

# Sonify the data, generate separate WAV files for each region, and save the plots
sonify_data(data)

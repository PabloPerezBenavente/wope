import numpy as np
import plotly.graph_objects as go
from ipywidgets import interact

# Function to simulate wavelength and its color
def wavelength_to_rgb(wavelength):
    """
    Convert wavelength in nanometers to an approximate RGB color.
    """
    if 380 <= wavelength <= 440:
        r = -(wavelength - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 < wavelength <= 490:
        r = 0.0
        g = (wavelength - 440) / (490 - 440)
        b = 1.0
    elif 490 < wavelength <= 510:
        r = 0.0
        g = 1.0
        b = -(wavelength - 510) / (510 - 490)
    elif 510 < wavelength <= 580:
        r = (wavelength - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 < wavelength <= 645:
        r = 1.0
        g = -(wavelength - 645) / (645 - 580)
        b = 0.0
    elif 645 < wavelength <= 780:
        r = 1.0
        g = 0.0
        b = 0.0
    else:
        r = g = b = 0.0  # Wavelength out of visible range

    # Gamma correction
    r = int((r**0.8) * 255)
    g = int((g**0.8) * 255)
    b = int((b**0.8) * 255)
    return f'rgb({r},{g},{b})'

# Function to create the graph
def plot_wavelength_graph(intensity=1, period=100):
    # Generate data
    wavelengths = np.linspace(380, 780, 200)
    colors = [wavelength_to_rgb(w) for w in wavelengths]
    amplitudes = intensity * np.sin(2 * np.pi * wavelengths / period)

    # Create the plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=wavelengths,
        y=amplitudes,
        mode='lines',
        line=dict(color='black', width=2),
        marker=dict(color=colors),
        name='Wavelength'
    ))

    fig.update_layout(
        title=f"Wavelength Graph (Intensity: {intensity}, Period: {period})",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Amplitude",
        template="plotly_dark"
    )

    fig.show()

# Interactive sliders for intensity and period
interact(plot_wavelength_graph, intensity=(0.1, 5.0, 0.1), period=(50, 300, 10))
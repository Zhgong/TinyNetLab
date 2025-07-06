import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.datasets import make_moons

# Deprecated: use moons_streamlit.py for a Streamlit interface
# Initial parameters
INITIAL_SAMPLES = 800
INITIAL_NOISE = 0.2

# Generate initial dataset
X, y = make_moons(n_samples=INITIAL_SAMPLES, noise=INITIAL_NOISE, random_state=42)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
scat = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=15)
ax.set_title("Moons 数据分布")

# Slider for noise
ax_noise = plt.axes([0.25, 0.1, 0.65, 0.03])
noise_slider = Slider(ax_noise, 'noise', 0.0, 0.5, valinit=INITIAL_NOISE, valstep=0.01)

# Slider for number of samples
ax_samples = plt.axes([0.25, 0.05, 0.65, 0.03])
samples_slider = Slider(ax_samples, 'samples', 100, 1000, valinit=INITIAL_SAMPLES, valstep=50)

def update(val):
    n_samples = int(samples_slider.val)
    noise = noise_slider.val
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    scat.set_offsets(X)
    scat.set_array(y)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

noise_slider.on_changed(update)
samples_slider.on_changed(update)

plt.show()

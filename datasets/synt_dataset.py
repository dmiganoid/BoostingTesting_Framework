%matplotlib qt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button


init_degree = 3
init_N = 1000
init_noise = 0


fig, ax = plt.subplots()
fig.subplots_adjust(left=0.25, bottom=0.25)

polynomial_degree_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
polynomial_degree_slider = Slider(
    ax=polynomial_degree_ax,
    label='Polynomial degree',
    valmin=1,
    valmax=10,
    valinit=init_degree,
    valstep=1
)

N_samples_ax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
N_samples_slider = Slider(
    ax=N_samples_ax,
    label='N samples',
    valmin=100,
    valmax=100000,
    valinit=init_N,
    valstep=1
)

noise_variance_ax = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
noise_variance_slider = Slider(
    ax=noise_variance_ax,
    label='Noise',
    valmin=0,
    valmax=1,
    valinit=init_noise,
    orientation="vertical"
)


polynomial = np.poly(np.random.uniform(-1, 1, init_degree))
polynomial /= abs(np.polyval(polynomial, np.linspace(-1,1,100))).max()

X = np.random.uniform(-1, 1, (init_N, 2))

curr_noise = init_noise
y = np.zeros(init_N)
for i in range(init_N):
    if X[i, 1] + np.random.normal(0, curr_noise) > np.polyval(polynomial, X[i, 0]):
        y[i] = 1



samples_scatter = ax.scatter(X[:, 0], X[:, 1], )
polynomial_line, = ax.plot(np.linspace(-1,1,100), np.polyval(polynomial, np.linspace(-1,1,100)), 'r--')
samples_scatter.set_color(c=np.where(y, 'blue', 'black'))


def update_polynomial(degree_val):
    global polynomial
    polynomial = np.poly(np.random.uniform(-1, 1, degree_val))
    polynomial /= abs(np.polyval(polynomial, np.linspace(-1,1,100))).max() * (-1 if np.random.random() < 0.4 else 1)
    polynomial_line.set_ydata(np.polyval(polynomial, np.linspace(-1,1,100)))
    update_noise(curr_noise)

def update_sample(N_val):
    global X
    X = np.random.uniform(-1, 1, (N_val, 2))
    samples_scatter.set_offsets(X)
    update_noise(curr_noise)

def update_noise(noise_val):
    global polynomial, curr_noise, y
    curr_noise = noise_val
    y = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if X[i, 1] + np.random.normal(0, curr_noise) > np.polyval(polynomial, X[i, 0]):
            y[i] = 1
    samples_scatter.set_color(c=np.where(y, 'blue', 'black'))
    fig.canvas.draw_idle()


save_ax = fig.add_axes([0.08, 0.16, 0.08, 0.04])
save_button = Button(save_ax, 'Save', hovercolor='0.975')


def save_sample(event):
    global X, y
    np.savetxt("dataset.csv", np.column_stack((X, y.astype(int))), delimiter=",")

generate_button_ax = fig.add_axes([0.85, 0.05, 0.12, 0.1])
generate_button = Button(generate_button_ax, 'Generate', hovercolor='0.975')


def gen_sample(event):
    global polynomial, X
    update_sample(X.shape[0])
    update_polynomial(polynomial.shape[0]-1)



generate_button.on_clicked(gen_sample)
save_button.on_clicked(save_sample)
polynomial_degree_slider.on_changed(update_polynomial)
N_samples_slider.on_changed(update_sample)
noise_variance_slider.on_changed(update_noise)


plt.show()
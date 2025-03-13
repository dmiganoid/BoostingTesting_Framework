import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from matplotlib.backend_bases import MouseButton
import os

init_degree = 2
init_N = 5
init_noise = 0


fig, ax = plt.subplots(figsize=(16, 9))
fig.subplots_adjust(left=0.25, bottom=0.25)
N_samples_ax = fig.add_axes([0.25, 0.1, 0.6, 0.03])
N_samples_slider = Slider(
    ax=N_samples_ax,
    label='N samples for control point',
    valmin=5,
    valmax=100,
    valinit=init_N,
    valstep=1,
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

bspline_deg_ax = fig.add_axes([0.0275, 0.25, 0.0225, 0.63])
bspline_deg_slider = Slider(
    ax=bspline_deg_ax,
    label='Deg',
    valmin=2,
    valmax=10,
    valinit=init_degree ,
    orientation="vertical",
    valstep=1
)

save_ax = fig.add_axes([0.08, 0.16, 0.08, 0.04])
fig.set_label("Dataset Generator")
save_button = Button(save_ax, 'Save', hovercolor='0.975')

generate_button_ax = fig.add_axes([0.55, 0.04, 0.12, 0.05])
generate_button = Button(generate_button_ax, 'Generate', hovercolor='0.975')

closed_button_ax = fig.add_axes([0.28, 0.04, 0.22, 0.05])
closed_button = CheckButtons(
    ax=closed_button_ax,
    labels=['Closed Spline'],
    actives=[False]
)

X_control_points = []
X = []
curr_noise = init_noise
bspline_degree = init_degree
labels = []
bspline_points = []
bspline_x = None
bspline_y = None
t_interval = None
bspline_closed = False
controlpoint_N = init_N

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.autoscale(enable=False)
X_control_points_scatter = ax.scatter([],[])
X_control_points_scatter.set_color('g')
bspline_points_scatter = ax.scatter([], [])
bspline_points_scatter.set_color('r')
bspline_line = ax.plot([], 'r--')
samples_scatter = ax.scatter([], [], )

def update_bsplinepoints(event):
    if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
        bspline_points.append([event.xdata, event.ydata])
        bspline_points_scatter.set_offsets(np.array(bspline_points))
    fig.canvas.draw_idle()

def update_bspline_c(event):
    global bspline_closed
    bspline_closed = not bspline_closed

def update_bspline_degree(val):
    global bspline_degree
    bspline_degree = val

def update_bspline():
    global bspline_x, bspline_y, bspline_points
    if bspline_x is not None and bspline_y is not None and len(bspline_points) == 0:
        if bspline_x.c[0] == bspline_x.c[-bspline_x.k]:
            bspline_points = [[bspline_x.c[i], bspline_y.c[i]] for i in range(len(bspline_x.c)-bspline_x.k)]
        else:
            bspline_points = [[bspline_x.c[i], bspline_y.c[i]] for i in range(len(bspline_x.c))]

    elif len(bspline_points) < bspline_degree + 1:
        bspline_points = []
        print("Not enough knots" if len(bspline_points) !=0 else "No knots")
        return False
    
    if bspline_closed:
        for k in range(bspline_degree):
            bspline_points.append(bspline_points[k])
        knots = np.linspace(0, 1, len(bspline_points) + bspline_degree+1)
    else:
        knots = np.concatenate([
            np.zeros(bspline_degree),
            np.linspace(0, 1, len(bspline_points) - bspline_degree+1),
            np.ones(bspline_degree)
        ])
    x, y = np.array(bspline_points)[:, 0], np.array(bspline_points)[:, 1]
    bspline_points = []
    try:
        bspline_x = BSpline(knots, x, bspline_degree, extrapolate=False)
        bspline_y = BSpline(knots, y, bspline_degree, extrapolate=False)
        return True
    except Exception as ex:
        print(ex)
        return False

def update_X_control_points(event):
    if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
        # Add clicked point to the list
        X_control_points.append([event.xdata, event.ydata])
        X_control_points_scatter.set_offsets(np.array(X_control_points))
    fig.canvas.draw_idle()
    
def update_X():
    global X, X_control_points
    if len(X_control_points) == 0:
        print("No X control points")
        return False
    X = []
    for x,y in X_control_points:
        for _ in range(controlpoint_N):
            X.append((x + np.random.normal(0, 0.25), y+np.random.normal(0, 0.25)))
    X_control_points = []
    X = np.array(X)
    return True

def mouse_click(event):
    if event.button is MouseButton.LEFT:
        update_bsplinepoints(event)
    elif event.button is MouseButton.RIGHT:
        update_X_control_points(event)

def update_sample(N_val):
    global controlpoint_N 
    controlpoint_N = N_val

def update_noise(noise_val):
    global curr_noise
    curr_noise = noise_val

def draw(event):
    global bspline_points, X, bspline_points_scatter, X_control_points_scatter, labels, t_interval
    if update_bspline():
        t_interval = (bspline_x.t[bspline_x.k], bspline_x.t[-bspline_x.k-1])
        bspline_line[0].set_xdata(bspline_x(np.linspace(*t_interval, 200, endpoint=True)))
        bspline_line[0].set_ydata(bspline_y(np.linspace(*t_interval, 200)))
    bspline_points_scatter.remove()
    bspline_points_scatter = ax.scatter([], [])

    if update_X():
        samples_scatter.set_offsets(X)
        X_control_points_scatter.remove()
        X_control_points_scatter = ax.scatter([], [])

    if bspline_x is not None and bspline_y is not None and len(X) != 0:
        labels = np.zeros(X.shape[0])
        dist = lambda t,x,y: (x-bspline_x(t))**2 + (y-bspline_y(t))**2
        for i in range(X.shape[0]):
            x, y = X[i]
            x += np.random.normal(0, curr_noise)
            y += np.random.normal(0, curr_noise)
            t0 = sum(t_interval)/2
            for t in np.linspace(*t_interval, 100):
                t0 = t if dist(t, x, y) < dist(t0,x,y) else t0
            t0 = minimize(dist, args=(x,y), x0=t0, bounds=[t_interval]).x[0]
            if t0 == t_interval[0]:
                t = minimize(dist, args=(x,y), x0=t_interval[1], bounds=[t_interval]).x[0]
                t0 = t if dist(t, x, y) <= dist(t0, x, y) else t0
            elif t0 == t_interval[1]:
                t = minimize(dist, args=(x,y), x0=t_interval[0], bounds=[t_interval]).x[0]
                t0 = t if dist(t, x, y) <= dist(t0, x, y) else t0
            labels[i] = (bspline_x.derivative()(t0)*(y-bspline_y(t0)) - bspline_y.derivative()(t0)* (x-bspline_x(t0))>=0)
            #ax.set_aspect('equal')
            #ax.plot((x, bspline_x(t0)), (y, bspline_y(t0)))
        
        samples_scatter.set_color(c=np.where(labels, 'blue', 'black'))
    fig.canvas.draw_idle()


def save_sample(event, path="datasets"):
    global X, labels
    i = 0
    while os.path.exists(f"{path}/dataset-generated-{i}.csv"):
        i+=1
    np.savetxt(f"{path}/dataset-generated-{i}.csv", np.column_stack((X, labels.astype(int))), delimiter=",")


generate_button.on_clicked(draw)
N_samples_slider.on_changed(update_sample)
bspline_deg_slider.on_changed(update_bspline_degree)
noise_variance_slider.on_changed(update_noise)
save_button.on_clicked(save_sample)
closed_button.on_clicked(update_bspline_c)
fig.canvas.mpl_connect('button_press_event', mouse_click)

plt.grid()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from matplotlib.backend_bases import MouseButton
import os

class DataSetGenerator:
    def __init__(self):
        init_degree = 2
        init_N = 5
        init_noise = 0


        self.fig, self.ax = plt.subplots(figsize=(16, 9))
        self.fig.subplots_adjust(left=0.25, bottom=0.25)
        self.N_samples_ax = self.fig.add_axes([0.25, 0.1, 0.6, 0.03])
        self.N_samples_slider = Slider(
            ax=self.N_samples_ax,
            label='N samples for control point',
            valmin=5,
            valmax=100,
            valinit=init_N,
            valstep=1,
        )

        self.noise_variance_ax = self.fig.add_axes([0.1, 0.25, 0.0225, 0.63])
        self.noise_variance_slider = Slider(
            ax=self.noise_variance_ax,
            label='Noise',
            valmin=0,
            valmax=1,
            valinit=init_noise,
            orientation="vertical"
        )

        self.bspline_deg_ax = self.fig.add_axes([0.0275, 0.25, 0.0225, 0.63])
        self.bspline_deg_slider = Slider(
            ax=self.bspline_deg_ax,
            label='Deg',
            valmin=2,
            valmax=10,
            valinit=init_degree ,
            orientation="vertical",
            valstep=1
        )

        self.save_ax = self.fig.add_axes([0.08, 0.16, 0.08, 0.04])
        self.fig.set_label("Dataset Generator")
        self.save_button = Button(self.save_ax, 'Save', hovercolor='0.975')

        self.generate_button_ax = self.fig.add_axes([0.55, 0.04, 0.12, 0.05])
        self.generate_button = Button(self.generate_button_ax, 'Generate', hovercolor='0.975')

        self.closed_button_ax = self.fig.add_axes([0.28, 0.04, 0.22, 0.05])
        self.closed_button = CheckButtons(
            ax=self.closed_button_ax,
            labels=['Closed Spline'],
            actives=[False]
        )

        self.X_control_points = []
        self.X = []
        self.curr_noise = init_noise
        self.bspline_degree = init_degree
        self.labels = []
        self.bspline_points = []
        self.bspline_x = None
        self.bspline_y = None
        self.t_interval = None
        self.bspline_closed = False
        self.controlpoint_N = init_N

        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.autoscale(enable=False)
        self.X_control_points_scatter = self.ax.scatter([],[])
        self.X_control_points_scatter.set_color('g')
        self.bspline_points_scatter = self.ax.scatter([], [])
        self.bspline_points_scatter.set_color('r')
        self.bspline_line = self.ax.plot([], 'r--')
        self.samples_scatter = self.ax.scatter([], [], )

        self.generate_button.on_clicked(self.draw)
        self.N_samples_slider.on_changed(self.update_sample)
        self.bspline_deg_slider.on_changed(self.update_bspline_degree)
        self.noise_variance_slider.on_changed(self.update_noise)
        self.save_button.on_clicked(self.save_sample)
        self.closed_button.on_clicked(self.update_bspline_c)
        self.fig.canvas.mpl_connect('button_press_event', self.mouse_click)
        plt.grid()
        plt.show()
    def update_bsplinepoints(self, event):
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.bspline_points.append([event.xdata, event.ydata])
            self.bspline_points_scatter.set_offsets(np.array(self.bspline_points))
        self.fig.canvas.draw_idle()

    def update_bspline_c(self, event):
        self.bspline_closed = not self.bspline_closed

    def update_bspline_degree(self,val):
        self.bspline_degree = val

    def update_bspline(self):
        if self.bspline_x is not None and self.bspline_y is not None and len(self.bspline_points) == 0:
            if self.bspline_x.c[0] == self.bspline_x.c[-self.bspline_x.k]:
                self.bspline_points = [[self.bspline_x.c[i], self.bspline_y.c[i]] for i in range(len(self.bspline_x.c)-self.bspline_x.k)]
            else:
                self.bspline_points = [[self.bspline_x.c[i], self.bspline_y.c[i]] for i in range(len(self.bspline_x.c))]

        elif len(self.bspline_points) < self.bspline_degree + 1:
            self.bspline_points = []
            print("Not enough knots" if len(self.bspline_points) !=0 else "No knots")
            return False
        
        if self.bspline_closed:
            for k in range(self.bspline_degree):
                self.bspline_points.append(self.bspline_points[k])
            knots = np.linspace(0, 1, len(self.bspline_points) + self.bspline_degree+1)
        else:
            knots = np.concatenate([
                np.zeros(self.bspline_degree),
                np.linspace(0, 1, len(self.bspline_points) - self.bspline_degree+1),
                np.ones(self.bspline_degree)
            ])
        x, y = np.array(self.bspline_points)[:, 0], np.array(self.bspline_points)[:, 1]
        self.bspline_points = []
        try:
            self.bspline_x = BSpline(knots, x, self.bspline_degree, extrapolate=False)
            self.bspline_y = BSpline(knots, y, self.bspline_degree, extrapolate=False)
            return True
        except Exception as ex:
            print(ex)
            return False

    def update_X_control_points(self, event):
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            # Add clicked point to the list
            self.X_control_points.append([event.xdata, event.ydata])
            self.X_control_points_scatter.set_offsets(np.array(self.X_control_points))
        self.fig.canvas.draw_idle()
        
    def update_X(self):
        if len(self.X_control_points) == 0:
            print("No X control points")
            return False
        self.X = []
        for x,y in self.X_control_points:
            for _ in range(self.controlpoint_N):
                self.X.append((x + np.random.normal(0, 0.25), y+np.random.normal(0, 0.25)))
        self.X_control_points = []
        self.X = np.array(self.X)
        return True

    def mouse_click(self, event):
        if event.button is MouseButton.LEFT:
            self.update_bsplinepoints(event)
        elif event.button is MouseButton.RIGHT:
            self.update_X_control_points(event)

    def update_sample(self, N_val):
        self.controlpoint_N = N_val

    def update_noise(self, noise_val):
        self.curr_noise = noise_val

    def draw(self, event):
        if self.update_bspline():
            self.t_interval = (self.bspline_x.t[self.bspline_x.k], self.bspline_x.t[-self.bspline_x.k-1])
            self.bspline_line[0].set_xdata(self.bspline_x(np.linspace(*self.t_interval, 200, endpoint=True)))
            self.bspline_line[0].set_ydata(self.bspline_y(np.linspace(*self.t_interval, 200)))
        self.bspline_points_scatter.remove()
        self.bspline_points_scatter = self.ax.scatter([], [])

        if self.update_X():
            self.samples_scatter.set_offsets(self.X)
            self.X_control_points_scatter.remove()
            self.X_control_points_scatter = self.ax.scatter([], [])

        if self.bspline_x is not None and self.bspline_y is not None and len(self.X) != 0:
            labels = np.zeros(self.X.shape[0])
            dist = lambda t,x,y: (x-self.bspline_x(t))**2 + (y-self.bspline_y(t))**2
            for i in range(self.X.shape[0]):
                x, y = self.X[i]
                x += np.random.normal(0, self.curr_noise)
                y += np.random.normal(0, self.curr_noise)
                t0 = sum(self.t_interval)/2
                for t in np.linspace(*self.t_interval, 100):
                    t0 = t if dist(t, x, y) < dist(t0,x,y) else t0
                t0 = minimize(dist, args=(x,y), x0=t0, bounds=[self.t_interval]).x[0]
                if t0 == self.t_interval[0]:
                    t = minimize(dist, args=(x,y), x0=self.t_interval[1], bounds=[self.t_interval]).x[0]
                    t0 = t if dist(t, x, y) <= dist(t0, x, y) else t0
                elif t0 == self.t_interval[1]:
                    t = minimize(dist, args=(x,y), x0=self.t_interval[0], bounds=[self.t_interval]).x[0]
                    t0 = t if dist(t, x, y) <= dist(t0, x, y) else t0
                labels[i] = (self.bspline_x.derivative()(t0)*(y-self.bspline_y(t0)) - self.bspline_y.derivative()(t0)* (x-self.bspline_x(t0))>=0)
                #ax.set_aspect('equal')
                #ax.plot((x, bspline_x(t0)), (y, bspline_y(t0)))
            
            self.samples_scatter.set_color(c=np.where(labels, 'blue', 'black'))
        self.fig.canvas.draw_idle()


    def save_sample(self, event, path="datasets"):
        i = 0
        while os.path.exists(f"{path}/dataset-generated-{i}.csv"):
            i+=1
        np.savetxt(f"{path}/dataset-generated-{i}.csv", np.column_stack((self.X, self.labels.astype(int))), delimiter=",")

if __name__=="__main__":
    DataSetGenerator()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class sim_mobile_robots:  # Visualizer on 2D plot

    # INITIALIZATION, run only once in the beginning
    # -----------------------------------------------------------------
    def __init__(self, modes=None):
        # Generate the simulation window and plot initial objects
        self.fig = plt.figure(1)
        self.ax = plt.gca()
        self.ax.set(xlabel="x [m]", ylabel="y [m]")
        self.ax.set_aspect('equal', adjustable='box', anchor='C')
        plt.tight_layout()
        # Plot initial value for trajectory and time stamp

        self.time_txt = self.ax.text(0.78, 0.01, 't = 0 s', color='k', fontsize='large',
                                     horizontalalignment='left', verticalalignment='bottom',
                                     transform=self.ax.transAxes)

        # Store the plot option (use icon or not)
        self.robot_num = len(modes)
        self.cluster = 0
        self.ellipse_patch = [None]
        self.draw_with_mobile_robot_icon = True
        self.traj_pl = [None] * self.robot_num
        self.icon_id = [None] * self.robot_num
        self.moro_patch = [None] * self.robot_num
        self.pos_pl = [None] * self.robot_num
        for i, mode in enumerate(modes):
            self.traj_pl[i], = self.ax.plot(0, 0, 'b--')
            if mode == 'omnidirectional':
                self.icon_id[i] = 3
            elif mode == 'unicycle':
                self.icon_id[i] = 2
            else:
                self.draw_with_mobile_robot_icon = False
            # Draw current robot position
            if self.draw_with_mobile_robot_icon:  # use mobile robot icon
                self.moro_patch[i] = None
                self.draw_icon(np.zeros(3), i)
            else:  # use simple x marker
                self.pos_pl[i], = self.ax.plot(0, 0, 'b', marker='X', markersize=10)

    def set_field(self, x_axis_range_tuple, y_axis_range_tuple):
        # set the plot limit with the given range
        self.ax.set(xlim=x_axis_range_tuple, ylim=y_axis_range_tuple)

    def show_goal(self, goal_state):
        # Draw the goal state as an arrow 
        # the arrow tail denotes the goal position 
        # the arrow direction denotes the goal direction / angle
        arrow_size = 0.2
        self.pl_goal = [None] * self.robot_num
        for i in range(self.robot_num):
            ar_d = [arrow_size * np.cos(goal_state[i * 3 + 2]), arrow_size * np.sin(goal_state[i * 3 + 2])]
            self.pl_goal[i] = plt.quiver(goal_state[i * 3], goal_state[i * 3 + 1], ar_d[0], ar_d[1],
                                  scale_units='xy', scale=1, color='r', width=0.1 * arrow_size)

    def show_formation(self, ellipse, width, height):
        self.cluster = int(len(ellipse) / 3)
        self.ellipse_patch = [None] * self.cluster
        for i in range(self.cluster):
            self.ellipse_patch[i] = patches.Ellipse((ellipse[3 * i], ellipse[3 * i + 1]), width, height, ellipse[3 * i + 2] * 180 / np.pi, fill=False)
            self.ax.add_artist(self.ellipse_patch[i])

    # PLOT UPDATES, run in every iterations
    # -----------------------------------------------------------------
    def update_formation(self, ellipse):
        for i in range(self.cluster):
            self.ellipse_patch[i].set_center((ellipse[3 * i], ellipse[3 * i + 1]))
            # print(ellipse[3 * i + 2])
            self.ellipse_patch[i].set_angle(ellipse[3 * i + 2] * 180 / np.pi)

    def update_time_stamp(self, float_current_time):
        # update the displayed time stamp
        self.time_txt.set_text('t = ' + f"{float_current_time:.1f}" + ' s')

    def update_goal(self, goal_state):
        # update the displayed goal state, especially when it is moving over time
        arrow_size = 0.2
        for i in range(self.robot_num):
            ar_d = [arrow_size * np.cos(goal_state[i * 3 + 2]), arrow_size * np.sin(goal_state[i * 3 + 2])]
            self.pl_goal[i].set_offsets([goal_state[i * 3], goal_state[i * 3 + 1]])
            self.pl_goal[i].set_UVC(ar_d[0], ar_d[1])

    def update_trajectory(self, state_historical_data):  # update robot status
        # Extract data for plotting
        for i in range(self.robot_num):
            trajectory_px = state_historical_data[:, 0 + i * 3]
            trajectory_py = state_historical_data[:, 1 + i * 3]
            robot_state = state_historical_data[-1, 3 * i: 3 * i + 3]
            # print(robot_state, state_historical_data)
            # Update the simulation with the new data
            self.traj_pl[i].set_data(trajectory_px, trajectory_py)  # plot trajectory
            if self.draw_with_mobile_robot_icon:  # use wheeled robot icon
                self.draw_icon(robot_state, i)
            else:  # update the x marker
                self.pos_pl[i].set_data(robot_state[0], robot_state[1])  # plot only last position
            # Pause to show the movement
        plt.pause(1e-2)

        # OPTIONAL PLOT, not necessary but provide nice view in simulation

    # -----------------------------------------------------------------
    def draw_icon(self, robot_state, index):  # draw mobile robot as an icon
        # Extract data for plotting
        px = robot_state[0]
        py = robot_state[1]
        th = robot_state[2]
        # Basic size parameter
        scale = 2
        body_rad = 0.08 * scale  # m
        wheel_size = [0.1 * scale, 0.02 * scale]
        arrow_size = body_rad
        # left and right wheels anchor position (bottom-left of rectangle)
        if self.icon_id[index] == 2:
            thWh = [th + 0., th + np.pi]  # unicycle
        else:
            thWh = [(th + i * (2 * np.pi / 3) - np.pi / 2) for i in range(3)]  # default to omnidirectional
        thWh_deg = [np.rad2deg(i) for i in thWh]
        wh_x = [px - body_rad * np.sin(i) - (wheel_size[0] / 2) * np.cos(i) + (wheel_size[1] / 2) * np.sin(i) for i in
                thWh]
        wh_y = [py + body_rad * np.cos(i) - (wheel_size[0] / 2) * np.sin(i) - (wheel_size[1] / 2) * np.cos(i) for i in
                thWh]
        # Arrow orientation anchor position
        ar_st = [px, py]  # [ px - (arrow_size/2)*np.cos(th), py - (arrow_size/2)*np.sin(th) ]
        ar_d = (arrow_size * np.cos(th), arrow_size * np.sin(th))
        # initialized unicycle icon at the center with theta = 0
        if self.moro_patch[index] is None:  # first time drawing
            self.moro_patch[index] = [None] * (2 + len(thWh))
            self.moro_patch[index][0] = self.ax.add_patch(plt.Circle((px, py), body_rad, color='#AAAAAAAA'))
            self.moro_patch[index][1] = plt.quiver(ar_st[0], ar_st[1], ar_d[0], ar_d[1],
                                            scale_units='xy', scale=1, color='b', width=0.1 * arrow_size)
            for i in range(len(thWh)):
                self.moro_patch[index][2 + i] = self.ax.add_patch(plt.Rectangle((wh_x[i], wh_y[i]),
                                                                         wheel_size[0], wheel_size[1],
                                                                         angle=thWh_deg[i], color='k'))
        else:  # update existing patch
            self.moro_patch[index][0].set(center=(px, py))
            self.moro_patch[index][1].set_offsets(ar_st)
            self.moro_patch[index][1].set_UVC(ar_d[0], ar_d[1])
            for i in range(len(thWh)):
                self.moro_patch[index][2 + i].set(xy=(wh_x[i], wh_y[i]))
                self.moro_patch[index][2 + i].angle = thWh_deg[i]

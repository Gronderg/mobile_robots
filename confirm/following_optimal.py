# Fundamental of Mobile Robot - AUT.710 - 2023
# Exercise 04 - Problem 01
# Hoang Pham, Nadun Ranasinghe, Long Phan

import cvxopt
import numpy as np
import matplotlib.pyplot as plt
from visualize_mobile_robots import sim_mobile_robots

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = 30 # total simulation duration in seconds
# Set initial state
RANDOM_INIT = True
init_state = np.array([[-1., 4., 0.], [4., 0., 0.], [-1., 2., 0.],
                       [4., 4., 0.], [3., 1., 0.], [4., -1., 0.]]).flatten()  # px, py, theta

IS_SHOWING_2DVISUALIZATION = True
ROBOT_NUM = 1
OBSTACLE_NUM = 6
OBSTACLES = np.zeros((OBSTACLE_NUM, 3))

SAFE_DISTANCE = 0.51
GAIN = 5.
GAMMA = 0.2
MAX_VEL = 2.
MAX_ROT = 5
Rsi = 0.5
b = 1
order = 3

# Define Field size for plotting (should be in tuple)
field_x = (-2, 5)
field_y = (-2, 5)


# IMPLEMENTATION FOR THE CONTROLLER
# ---------------------------------------------------------------------
def compute_control_input(desired_state, forecasted_desired_state, robot_state):
    # Check if using static gain
    # print(robot_state)

    current_input = saturate_velocity(GAIN * (desired_state - robot_state) + (forecasted_desired_state - desired_state)/Ts)


    for i in range(ROBOT_NUM):
        h = np.zeros(OBSTACLE_NUM)
        H = np.zeros((OBSTACLE_NUM, 2))

        for j in range(OBSTACLE_NUM):

            obs_distance = calculate_distance(robot_state[3 * i: 3 * i + 2], OBSTACLES[j, 0:2])
            h[j] = obs_distance ** 2 - Rsi ** 2
            H[j, :] = 2 * (robot_state[3 * i: 3 * i + 2] - OBSTACLES[j, 0:2])


        # Gamma calculation
        h = b * (h ** order)

        # Regulated u_gtg
        u_gtg = current_input[3 * i: 3 * i + 2]

        # Construct Q, H, b, c for QP-based controller
        Q_mat = 2 * cvxopt.matrix(np.eye(2), tc='d')
        c_mat = -2 * cvxopt.matrix(u_gtg, tc='d')

        H_mat = cvxopt.matrix(-H, tc='d')
        b_mat = cvxopt.matrix(h, tc='d')

        # Find u*
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(Q_mat, c_mat, H_mat, b_mat, verbose=False)

        current_input[3 * i: 3 * i + 2] = np.array([sol['x'][0], sol['x'][1]])

    return saturate_velocity(current_input)


def calculate_distance(former, latter):
    # Calculate Euclidean distance
    distance = np.sqrt((former[0] - latter[0]) ** 2 + (former[1] - latter[1]) ** 2)
    return distance.item()


def random_initial_state():
    random_state = np.zeros(ROBOT_NUM * 3)
    range_x = field_x[1] - field_x[0]
    range_y = field_y[1] - field_y[0]
    range_theta = 2 * np.pi

    for i in range(ROBOT_NUM):
        random_state[3 * i] = round(np.random.rand() * range_x + field_x[0], 4)
        random_state[3 * i + 1] = round(np.random.rand() * range_y + field_y[0], 4)
        random_state[3 * i + 2] = round(np.random.rand() * range_theta - np.pi, 4)

    return random_state


def make_trajectory(target_num):
    desired_trajectory = np.zeros((target_num, 3))

    mean_x = (field_x[0] + field_x[1]) / 2
    mean_y = (field_y[0] + field_y[1]) / 2

    for i in range(target_num):
        desired_theta = (i * Ts + np.pi / 2) % (2 * np.pi) - np.pi
        desired_trajectory[i] = np.array([mean_x - 2 * np.cos(i * Ts), mean_y - 2 * np.sin(i * Ts), desired_theta])

    return desired_trajectory


def generate_obstacle():
    range_x = field_x[1] - field_x[0] / 2
    range_y = field_y[1] - field_y[0] / 2
    for i in range(OBSTACLE_NUM):
        OBSTACLES[i, 0] = round(np.random.rand() * range_x + field_x[0] / 2, 4)
        OBSTACLES[i, 1] = round(np.random.rand() * range_y + field_y[0] / 2, 4)
    print(OBSTACLES)


def saturate_velocity(velocity):
    # Regulate the control input to comply with robot constraints
    for i in range(ROBOT_NUM):
        linear_velocity = np.sqrt(velocity[3 * i] ** 2 + velocity[3 * i + 1] ** 2)

        if linear_velocity > MAX_VEL:
            velocity[3 * i] /= linear_velocity / MAX_VEL
            velocity[3 * i + 1] /= linear_velocity / MAX_VEL

        velocity[3 * i + 2] = velocity[3 * i + 2] if abs(velocity[3 * i + 2]) <= MAX_ROT else np.sign(velocity[3 * i + 2]) * MAX_ROT
        velocity[3 * i + 2] = ((velocity[3 * i + 2] + np.pi) % (2 * np.pi)) - np.pi

    return velocity


# MAIN SIMULATION COMPUTATION
# ---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max / Ts)  # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    if RANDOM_INIT:
        robot_state = random_initial_state()
    else:
        robot_state = init_state.copy()  # numpy array for [px, py, theta]
    # desired_state = np.array([-2., 1., 0.])  # numpy array for goal / the desired [px, py, theta]
    desired_trajectory = make_trajectory(sim_iter + 1)
    desired_state = desired_trajectory[0]
    generate_obstacle()

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    goal_history = np.zeros((sim_iter, len(desired_state)))
    input_history = np.zeros((sim_iter, 3 * ROBOT_NUM))  # for [vx, vy, omega] vs iteration time

    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_input = ['omnidirectional'] * ROBOT_NUM
        sim_visualizer = sim_mobile_robots(sim_input)  # Omnidirectional Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area
        sim_visualizer.show_goal(desired_state)
        for i in range(OBSTACLE_NUM):
            sim_visualizer.ax.add_patch(plt.Circle(OBSTACLES[i], 0.3, color='r'))
            sim_visualizer.ax.add_patch(plt.Circle(OBSTACLES[i], SAFE_DISTANCE, color='r', fill=False))

    for it in range(sim_iter):
        current_time = it * Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # Compute control input
        current_input = compute_control_input(desired_state, desired_trajectory[it + 1], robot_state)

        # record the computed input at time-step t
        input_history[it] = current_input

        if IS_SHOWING_2DVISUALIZATION:  # Update Plot
            sim_visualizer.update_time_stamp(current_time)
            sim_visualizer.update_goal(desired_state)
            sim_visualizer.update_trajectory(state_history[:it + 1])  # up to the latest data

        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        # for i in range(ROBOT_NUM):
        robot_state = robot_state + Ts * current_input  # will be used in the next iteration
        for i in range(ROBOT_NUM):
            robot_state[i * 3 + 2] = ((robot_state[i * 3 + 2] + np.pi) % (2 * np.pi)) - np.pi  # ensure theta within [-pi pi]
        desired_state = desired_trajectory[it + 1]

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, input_history


if __name__ == '__main__':
    # Call main computation for robot simulation
    state_history, input_history = simulate_control()
    # ADDITIONAL PLOTTING
    # ----------------------------------------------
    t = [i * Ts for i in range(round(t_max / Ts))]

    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    for i in range(ROBOT_NUM):
        ax.plot(t, input_history[:, i * 3], label=f'vx{i + 1} [m/s]')
        ax.plot(t, input_history[:, 1 + i * 3], label=f'vy{i + 1} [m/s]')
        ax.plot(t, input_history[:, 2 + i * 3], label=f'omega{i + 1} [rad/s]')

    ax.set(xlabel="t [s]", ylabel="control input")
    plt.legend()
    plt.grid()

    # Plot historical data of state
    fig3 = plt.figure(3)
    ax = plt.gca()
    for i in range(ROBOT_NUM):
        ax.plot(t, state_history[:, i * 3], label=f'px{i + 1} [m]')
        ax.plot(t, state_history[:, 1 + i * 3], label=f'py{i + 1} [m]')
        ax.plot(t, state_history[:, 2 + i * 3], label=f'theta{i + 1} [rad]')

    ax.set(xlabel="t [s]", ylabel="state")
    plt.legend()
    plt.grid()

    plt.show()
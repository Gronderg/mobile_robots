import cvxopt
import numpy as np
import matplotlib.pyplot as plt
from visualize_mobile_robots import sim_mobile_robots
import ctypes 

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = 40 # total simulation duration in seconds
# Set initial state
RANDOM_INIT = False
# init_state = np.array([[-1., 4., 0.], [4., 0., 0.], [-1., 2., 0.],
#                        [4., 4., 0.], [3., 1., 0.], [4., -1., 0.]]).flatten()  # px, py, theta
init_state = np.array([[-1., 4., -np.pi], [4., 0., 3*np.pi/2], [-1., 2., -np.pi/2], [4., 4., np.pi/3]]).flatten()  # px, py, theta
# init_state = np.array([[1., 1., 0.], [1., 1.6, 0.], [1., 2.2, 0.], [1., 2.8, 0.]]).flatten()  # px, py, theta


IS_SHOWING_2DVISUALIZATION = True
ROBOT_NUM = 4
# LAPLACIAN_MAT = np.array([[2., -1., 0., 0., 0., -1.],
#                           [-1., 2., -1., 0., 0., 0.],
#                           [0., -1., 2., -1., 0., 0.],
#                           [0., 0., -1., 2., -1., 0.],
#                           [0., 0., 0., -1., 2., -1.],
#                           [-1., 0., 0., 0., -1., 2.]])
LAPLACIAN_MAT = np.array([[2., -1., 0., -1.],
                          [-1., 2., -1., 0.],
                          [0., -1., 2., -1.],
                          [-1., 0., -1., 2.]])

# B_MAT = np.array([-0.5, 0.87, 0.,
#                   0.5, 0.87, 0.,
#                   1., 0., 0.,
#                   0.5, -0.87, 0.,
#                   -0.5, -0.87, 0.,
#                   -1., 0., 0.]) * 2
B_MAT = np.array([-1., -1., 0., -1., 1., 0., 1., 1., 0., 1., -1., 0.])
# B_MAT = np.array([2., -1., 0., 0., 0., 0., 0., 0., 0., 2., 1., 0.])
GAMMA = 0.2

Rsi = 0.5
b = 10
order = 3
# spec of unicycle
l = 0.06
ROBOT_RADIUS = 0.08
WHEEL_RADIUS = 0.066/2
# for unicycle
MAX_ROT_SPEED = 2.84
MAX_TRANS_SPEED = 0.22
# for omni caster 
MAX_VEL = 0.5
MAX_ROT = 5.


# Define Field size for plotting (should be in tuple)
field_x = (-2, 5)
field_y = (-2, 5)


# IMPLEMENTATION FOR THE CONTROLLER
# ---------------------------------------------------------------------
def compute_control_input(robot_state):
    # Check if using static gain
    # print(robot_state)

    P_mat_1 = np.zeros((ROBOT_NUM, ROBOT_NUM))
    P_mat_2 = np.eye(ROBOT_NUM, ROBOT_NUM)
    P_mat_3 = -LAPLACIAN_MAT
    P_mat_4 = -GAMMA * LAPLACIAN_MAT
    P_mat = np.block([[P_mat_1, P_mat_2], [P_mat_3, P_mat_4]])
    B_mat = np.block([np.zeros(ROBOT_NUM * 3), B_MAT])

    new_state = np.kron(P_mat, np.eye(3, 3)) @ robot_state.flatten() + B_mat
    # print(np.kron(P_mat, np.eye(3, 3)) @ robot_state.flatten())
    # print(new_state)
    # print("---------------------------------------------------")
    current_input = saturate_velocity(new_state[3 * ROBOT_NUM:])
    temporary_input = saturate_velocity(new_state[3 * ROBOT_NUM:])

    for i in range(ROBOT_NUM):
        h = np.zeros(ROBOT_NUM)
        d = np.zeros(ROBOT_NUM)
        H = np.zeros((ROBOT_NUM, 2))
        obs_distances = np.zeros(ROBOT_NUM)

        for j in range(ROBOT_NUM):
            if i == j:
                continue

            obs_distance = calculate_distance(robot_state[3 * i: 3 * i + 2], robot_state[3 * j: 3 * j + 2])
            h[j] = obs_distance ** 2 - Rsi ** 2
            H[j, :] = 2 * (robot_state[3 * i: 3 * i + 2] - robot_state[3 * j: 3 * j + 2])
            d[j] = 2 * (robot_state[3 * i: 3 * i + 2] - robot_state[3 * j: 3 * j + 2]) @ temporary_input[3 * j: 3 * j + 2]
            obs_distances[j] = obs_distance

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

def caster_transform(robot_state):
    uni_state = np.zeros(3*ROBOT_NUM)
    for i in range(ROBOT_NUM):
        idx = 3*i
        x_agent = robot_state[idx:(idx+2)]
        theta = robot_state[idx+2]
        S = x_agent + np.array([l*np.cos(theta), l*np.sin(theta)])
        
        uni_state[idx:(idx+2)] = S
        uni_state[idx+2] = robot_state[idx+2]
    return uni_state

def control_transform(input, robot_state):
    current_input = np.zeros(8)
    for i in range(ROBOT_NUM):
        theta = robot_state[i * 3 + 2]
        current_input[2*i:2*i+2] = np.array([[1, 0], [0, 1/l]])@np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])@input[3*i:3*i+2]
    
    return current_input

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

def saturate_velocity(velocity):
    # Regulate the control input to comply with robot constraints
    for i in range(ROBOT_NUM):
        linear_velocity = np.sqrt(velocity[3 * i] ** 2 + velocity[3 * i + 1] ** 2)

        if linear_velocity > MAX_VEL:
            velocity[3 * i] /= linear_velocity / MAX_VEL
            velocity[3 * i + 1] /= linear_velocity / MAX_VEL

        velocity[3 * i + 2] = velocity[3 * i + 2] if abs(velocity[3 * i + 2]) <= MAX_ROT else np.sign(velocity[3 * i + 2]) * MAX_ROT

    return velocity

def check_velocity(velocity):
    # Regulate the control input to comply with robot constraints
    w_r = (velocity[0] + ROBOT_RADIUS*velocity[1])/WHEEL_RADIUS
    w_l = (velocity[0] - ROBOT_RADIUS*velocity[1])/WHEEL_RADIUS
    if w_r > MAX_ROT_SPEED:
        # ctypes.windll.user32.MessageBoxW(0, "Right", "Warming", 1)
        w_r = MAX_ROT_SPEED
        velocity[0] = ROBOT_RADIUS*(w_r+w_l)/2
        velocity[1] = ROBOT_RADIUS*(w_r-w_l)/(2*WHEEL_RADIUS)
    if w_l > MAX_ROT_SPEED:
        # ctypes.windll.user32.MessageBoxW(0, "Left", "Warming", 1)
        w_l = MAX_ROT_SPEED
        velocity[0] = ROBOT_RADIUS*(w_r+w_l)/2
        velocity[1] = ROBOT_RADIUS*(w_r-w_l)/(2*WHEEL_RADIUS)
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

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    input_history = np.zeros((sim_iter, 3 * ROBOT_NUM))  # for [vx, vy, omega] vs iteration time
    current_input_history = np.zeros((sim_iter, 2 * ROBOT_NUM))
    right_history = np.zeros((sim_iter,1))
    left_history = np.zeros((sim_iter,1))
    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_input = ['unicycle'] * ROBOT_NUM
        sim_visualizer = sim_mobile_robots(sim_input)  # Omnidirectional Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area

    for it in range(sim_iter):
        current_time = it * Ts
        # record current state at time-step t
        state_history[it] = robot_state
        uni_state = caster_transform(robot_state)

        # Compute control input
        input = compute_control_input(np.concatenate((uni_state, input_history[it-1])))
        input_history[it] = input

        current_input = control_transform(input, robot_state)
        print(current_input)
        # for i in range(ROBOT_NUM):

        #     if current_input[2*i] > MAX_TRANS_SPEED:
        #         ctypes.windll.user32.MessageBoxW(0, "v", "Warming", 1)
        #     if current_input[2*i+1] > MAX_ROT_SPEED:
        #         ctypes.windll.user32.MessageBoxW(0, "omega", "Warming", 1)
        w_r = (current_input[0] + ROBOT_RADIUS*current_input[1])/WHEEL_RADIUS
        w_l = (current_input[0] - ROBOT_RADIUS*current_input[1])/WHEEL_RADIUS
        right_history[it] = w_r
        left_history[it] =w_l

        # record the computed input at time-step t
        current_input_history[it] = current_input
        
        if IS_SHOWING_2DVISUALIZATION:  # Update Plot
            sim_visualizer.update_time_stamp(current_time)
            sim_visualizer.update_trajectory(state_history[:it + 1])  # up to the latest data

        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        # for i in range(ROBOT_NUM):
        for i in range(ROBOT_NUM):
            idx = 3*i
            theta = robot_state[idx + 2]
            angle_ = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
            robot_state[idx:(idx+3)] = robot_state[idx:(idx+3)] + Ts*(angle_ @ current_input[2*i:2*i+2]) # will be used in the next iteration
            robot_state[idx+2] = ( (robot_state[idx+2] + np.pi) % (2*np.pi) ) - np.pi # ensure theta within [-pi pi]

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, current_input_history, right_history, left_history


if __name__ == '__main__':
    # Call main computation for robot simulation
    state_history, input_history, right, left = simulate_control()
    # ADDITIONAL PLOTTING
    # ----------------------------------------------
    t = [i * Ts for i in range(round(t_max / Ts))]

    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    for i in range(ROBOT_NUM):
        ax.plot(t, input_history[:, i * 2], label=f'v{i + 1} [m/s]')
        # ax.plot(t, input_history[:, 1 + i * 2], label=f'omega{i + 1} [rad/s]')
        # ax.plot(t, input_history[:, 2 + i * 3], label=f'omega{i + 1} [rad/s]')

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

    # Plot historical data of control input
    fig4 = plt.figure(4)
    ax = plt.gca()

    ax.plot(t, right[:,:], label=f'right [rad/s]')
    ax.plot(t, left[:, :], label=f'left [rad/s]')


    ax.set(xlabel="t [s]", ylabel="rad/s")
    plt.legend()
    plt.grid()

    # Plot historical data of control input
    fig5 = plt.figure(5)
    ax = plt.gca()
    for i in range(ROBOT_NUM):
        # ax.plot(t, input_history[:, i * 2], label=f'v{i + 1} [m/s]')
        ax.plot(t, input_history[:, 1 + i * 2], label=f'omega{i + 1} [rad/s]')
        # ax.plot(t, input_history[:, 2 + i * 3], label=f'omega{i + 1} [rad/s]')

    ax.set(xlabel="t [s]", ylabel="control input")
    plt.legend()
    plt.grid()

    plt.show()
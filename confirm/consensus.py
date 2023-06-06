import numpy as np
import matplotlib.pyplot as plt
from visualize_mobile_robots import sim_mobile_robots

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = 5 # total simulation duration in seconds
# Set initial state
init_state = np.array([[-1., 4., 0.], [4., 0., 0.], [-1., 2., 0.], [4., 4., 0.]]).flatten()  # px, py, theta

IS_SHOWING_2DVISUALIZATION = True
ROBOT_NUM = 4
LAPLACIAN_MAT = np.array([[2., -1., 0., -1.],
                          [-1., 2., -1., 0.],
                          [0., -1., 2., -1.],
                          [-1., 0., -1., 2.]])
B_MAT = np.array([-1., -0.5, 0., -1., 0.5, 0., 1., 0.5, 0., 1., -0.5, 0.])
GAMMA = 0.2
l = 0.06
ROBOT_RADIUS = 0.08
WHEEL_RADIUS = 0.066/2
MAX_ROT_SPEED = 2.84

MAX_VEL = 0.05 # WHEEL_RADIUS*MAX_ROT_SPEED
# MAX_ROT = MAX_VEL/(2*ROBOT_RADIUS)
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
    # print(np.kron(P_mat, np.eye(3, 3)))
    new_state = np.kron(P_mat, np.eye(3, 3)) @ robot_state.flatten() + B_mat
    current_input = new_state[3 * ROBOT_NUM:]
    # print(current_input)
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

def saturate_velocity(velocity):
    # Regulate the control input to comply with robot constraints
    for i in range(ROBOT_NUM):
        linear_velocity = np.sqrt(velocity[3 * i] ** 2 + velocity[3 * i + 1] ** 2)

        if linear_velocity > MAX_VEL:
            velocity[3 * i] /= linear_velocity / MAX_VEL
            velocity[3 * i + 1] /= linear_velocity / MAX_VEL

        # velocity[3 * i + 2] = velocity[3 * i + 2] if abs(velocity[3 * i + 2]) <= MAX_ROT else np.sign(velocity[3 * i + 2]) * MAX_ROT

    return velocity

# MAIN SIMULATION COMPUTATION
# ---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max / Ts)  # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy()  # numpy array for [px, py, theta]
    # desired_state = np.array([-2., 1., 0.])  # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    input_history = np.zeros((sim_iter, 3 * ROBOT_NUM))  # for [vx, vy, omega] vs iteration time
    current_input_history = np.zeros((sim_iter, 2 * ROBOT_NUM))
    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_visualizer = sim_mobile_robots(['unicycle']*ROBOT_NUM)  # Omnidirectional Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area

    for it in range(sim_iter):
        current_time = it * Ts
        # record current state at time-step t
        state_history[it] = robot_state
        uni_state = caster_transform(robot_state)
        # Compute control input
        input = compute_control_input(np.concatenate((uni_state, input_history[it-1])))
        input_history[it] = input
        # record the computed input at time-step t
        current_input = control_transform(input, robot_state)
        current_input_history[it] = current_input

        if IS_SHOWING_2DVISUALIZATION:  # Update Plot
            sim_visualizer.update_time_stamp(current_time)
            sim_visualizer.update_trajectory(state_history[:it + 1])  # up to the latest data

        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        for i in range(ROBOT_NUM):
            idx = 3*i
            theta = robot_state[idx + 2]
            angle_ = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
            robot_state[idx:(idx+3)] = robot_state[idx:(idx+3)] + Ts*(angle_ @ current_input[2*i:2*i+2]) # will be used in the next iteration
            robot_state[idx+2] = ( (robot_state[idx+2] + np.pi) % (2*np.pi) ) - np.pi # ensure theta within [-pi pi]

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, current_input_history


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
        ax.plot(t, input_history[:, i * 2], label=f'v{i + 1} [m/s]')
        ax.plot(t, input_history[:, 1 + i * 2], label=f'omega{i + 1} [rad/s]')
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

    plt.show()
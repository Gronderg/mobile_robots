import numpy as np
import matplotlib.pyplot as plt
from visualize_mobile_robots import sim_mobile_robots
import cvxopt
import ctypes 
# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = 50 # total simulation duration in seconds
# Set initial state
# d_short = np.sqrt(0.5)
d_ = 1.

# init_state = np.array([[1., 2., 0.], [(1.+d_short), 2., 0.], [(1.+d_short), (2.0 - d_short), 0.], [1., (2.0-d_short), 0.]]).flatten()
# goal_state = np.array([[5., 5., 0.], [5.+d_short, 5., 0.], [5.+d_short, 5.-d_short, 0.], [5., 5.-d_short, 0.]]).flatten()
ini = 2.5
init_state = np.array([[1.5, ini+d_/2., 0.], [2.5, ini+d_/2, 0.], [2.5, ini-d_/2, 0.], [1.5, ini-d_/2, 0.]]).flatten()
goal = 5.
goal_state = np.array([[goal-d_/2, 3.5, 0.], [goal+d_/2, 3.5, 0.], [goal+d_/2, 2.5, 0.], [goal-d_/2, 2.5, 0.]]).flatten()


eps = 0.1
d_safe = 0.25
eps_obst = 0.02

b = 10
order = 3

l = 0.06
ROBOT_RADIUS = 0.08
WHEEL_RADIUS = 0.066/2
# for unicycle
MAX_ROT_SPEED = 2.84
MAX_TRANS_SPEED = 0.22
# for omni caster 
MAX_VEL = 0.1
MAX_ROT = 5.
K = 1.5
BETA = 1.


IS_SHOWING_2DVISUALIZATION = True
ROBOT_NUM = 4
obstacle1 = np.array([3.5, 2., 0.])
obstacle2 = np.array([3.5, 3.7, 0.])

# Define Field size for plotting (should be in tuple)
field_x = (0, 8)
field_y = (0, 8)


# IMPLEMENTATION FOR THE CONTROLLER
# ---------------------------------------------------------------------
def compute_control_input(robot_state, goal_agent, h_safe):
    current_input = np.zeros(ROBOT_NUM * 3)
    u_gtg = gtg_control_input(goal_agent, robot_state)
    
    for i in range(ROBOT_NUM):
        
        h = np.zeros((2*ROBOT_NUM + 2,1))
        # d = np.zeros(ROBOT_NUM)
        H = np.zeros((2*ROBOT_NUM + 2, 2))
        # obs_distances = np.zeros(ROBOT_NUM)
        static_obs_distance1 = calculate_distance(robot_state[3 * i: 3 * i + 2], obstacle1[:2])
        static_obs_distance2 = calculate_distance(robot_state[3 * i: 3 * i + 2], obstacle2[:2])
        h_safe[2*i] = static_obs_distance1
        h_safe[2*i + 1] = static_obs_distance2
        for j in range(ROBOT_NUM):
            d_ = 1.
            if i != j:            
                obs_distance = calculate_distance(robot_state[3 * i: 3 * i + 2], robot_state[3 * j: 3 * j + 2])
                if (i - j == 2) or(j-i ==2):
                    d_ = np.sqrt(2.)
                H[2*j, :] = -2 * (robot_state[3 * i: 3 * i + 2] - robot_state[3 * j: 3 * j + 2]).T
                H[2*j + 1, :] = 2 * (robot_state[3 * i: 3 * i + 2] - robot_state[3 * j: 3 * j + 2]).T
                # d[j] = 2 * (robot_state[3 * i: 3 * i + 2] - robot_state[3 * j: 3 * j + 2]).T @ temporary_input[3 * j: 3 * j + 2]
                # h[j] = b * ((obs_distance ** 2 - Rsi ** 2) ** order) - d[j]
                h[2*j,:] = b * ((obs_distance ** 2 - (d_ - eps) ** 2) ** order)
                h[2*j + 1,:] = b * (((d_ + eps) ** 2 - obs_distance ** 2) ** order)
                # obs_distances[j] = obs_distance
        H[2*ROBOT_NUM, :] = -2*(robot_state[3 * i: 3 * i + 2] - obstacle1[:2]).T
        H[2*ROBOT_NUM + 1, :] = -2*(robot_state[3 * i: 3 * i + 2] - obstacle2[:2]).T
        h[2*ROBOT_NUM,:] = b*(static_obs_distance1**2 - d_safe**2)**order
        h[2*ROBOT_NUM + 1,:] = b*(static_obs_distance2
                                  **2 - d_safe**2)**order
        # Construct Q, H, b, c for QP-based controller
        Q_mat = 2 * cvxopt.matrix(np.eye(2), tc='d')
        c_mat = -2 * cvxopt.matrix(u_gtg[3 * i: 3 * i + 2], tc='d')

        H_mat = cvxopt.matrix(H, tc='d')
        b_mat = cvxopt.matrix(h, tc='d')

        # Find u*
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(Q_mat, c_mat, H_mat, b_mat, verbose=False)

        current_input[3 * i: 3 * i + 2] = np.array([sol['x'][0], sol['x'][1]])

    return saturate_velocity(current_input), h_safe

def gtg_control_input(desired_state, robot_state):
    # Feel free to adjust the input and output of the function as needed.
    # And make sure it is reflected inside the loop in simulate_control()
    current_input = np.zeros(ROBOT_NUM * 3)
    for i in range(ROBOT_NUM):
        distance = calculate_distance(robot_state[3 * i:3 * i + 2], desired_state[3 * i:3 * i + 2])
        GAIN = K * (1 - np.exp(-BETA * distance)) / distance
        current_input[3 * i:3 * i + 3] = GAIN * (desired_state[3 * i:3 * i + 3] - robot_state[3 * i:3 * i + 3])
    return current_input

def calculate_distance(former, latter):
    # Calculate Euclidean distance
    distance = np.sqrt((former[0] - latter[0]) ** 2 + (former[1] - latter[1]) ** 2)
    return distance.item()

def saturate_velocity(velocity):
    # Regulate the control input to comply with robot constraints
    for i in range(ROBOT_NUM):
        linear_velocity = np.sqrt(velocity[3 * i] ** 2 + velocity[3 * i + 1] ** 2)

        if linear_velocity > MAX_VEL:
            velocity[3 * i] /= linear_velocity / MAX_VEL
            velocity[3 * i + 1] /= linear_velocity / MAX_VEL

        velocity[3 * i + 2] = velocity[3 * i + 2] if abs(velocity[3 * i + 2]) <= MAX_ROT else np.sign(velocity[3 * i + 2]) * MAX_ROT

    return velocity

def caster_transform(robot_state):
    uni_state = np.zeros(3*ROBOT_NUM)
    for i in range(ROBOT_NUM):
        idx = 3*i
        x_agent = robot_state[idx:(idx+2)]
        theta = robot_state[idx+2]
        S = x_agent + np.array([l*np.cos(theta), l*np.sin(theta)])
        
        uni_state[idx:(idx+2)] = S
        uni_state[idx+2] = theta
    return uni_state

def control_transform(input, robot_state):
    current_input = np.zeros(2*ROBOT_NUM)
    for i in range(ROBOT_NUM):
        theta = robot_state[i * 3 + 2]
        current_input[2*i:2*i+2] = np.array([[1, 0], [0, 1/l]])@np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])@input[3*i:3*i+2]
    
    return current_input

# MAIN SIMULATION COMPUTATION
# ---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max / Ts)  # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy()  # numpy array for [px, py, theta]
    desired_state = goal_state.copy()
    goal_history = np.zeros( (sim_iter, len(desired_state)) )

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    input_history = np.zeros((sim_iter, 2 * ROBOT_NUM))  # for [vx, vy, omega] vs iteration time
    h_safe_history = np.zeros((sim_iter, 2 * ROBOT_NUM))

    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_input = ['unicycle'] * ROBOT_NUM
        sim_visualizer = sim_mobile_robots(sim_input) 
        sim_visualizer.set_field(field_x, field_y)  # set plot area

        sim_visualizer.ax.add_patch(plt.Circle((3.5,2.),0.2,color='r'))
        sim_visualizer.ax.add_patch(plt.Circle((3.5,2.),d_safe,color='r',fill = False))
        sim_visualizer.ax.add_patch(plt.Circle((3.5,2.),d_safe+eps_obst,color='g', fill=False))
       
        sim_visualizer.ax.add_patch(plt.Circle((3.5,3.7),0.2,color='r'))
        sim_visualizer.ax.add_patch(plt.Circle((3.5,3.7),d_safe,color='r',fill = False))
        sim_visualizer.ax.add_patch(plt.Circle((3.5,3.7),d_safe+eps_obst,color='g', fill=False))



    for it in range(sim_iter):
        current_time = it * Ts
        # record current state at time-step t
        state_history[it,:] = robot_state
        goal_history[it,:] = desired_state
        caster_state = caster_transform(robot_state)
        h_safe = np.zeros(2*ROBOT_NUM)

        # Compute control input
        input, h_safe = compute_control_input(caster_state, desired_state, h_safe)
        print(input)
        print("---------------------------------------------")
        current_input = control_transform(input, robot_state)

        for i in range(ROBOT_NUM):
            if current_input[2*i] > MAX_TRANS_SPEED:
                ctypes.windll.user32.MessageBoxW(0, "v", "Warming", 1)
            if current_input[2*i+1] > MAX_ROT_SPEED:
                ctypes.windll.user32.MessageBoxW(0, "omega", "Warming", 1)
    
        # record the computed input at time-step t
        input_history[it, :] = current_input
        h_safe_history[it,:] = h_safe

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
    return state_history, input_history, h_safe_history


if __name__ == '__main__':
    # Call main computation for robot simulation
    state_history, input_history, h_safe_history = simulate_control()
    # ADDITIONAL PLOTTING
    # ----------------------------------------------
    t = [i * Ts for i in range(round(t_max / Ts))]

    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    for i in range(ROBOT_NUM):
        ax.plot(t, input_history[:, i * 2], label=f'v{i + 1} [m/s]')
        ax.plot(t, input_history[:, 1 + i * 2], label=f'omega{i + 1} [rad/s]')
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

    # Plot historical data of state
    fig4 = plt.figure(4)
    ax = plt.gca()
    for i in range(ROBOT_NUM):
        ax.plot(t, h_safe_history[:, i * 2], label=f'distance: robot{i + 1} and obs1 [m]')
        ax.plot(t, h_safe_history[:, 1 + i * 2], label=f'distance: robot{i + 1} and obs2 [m]')
    ax.set(xlabel="t [s]", ylabel="[m]")
    plt.legend()
    plt.grid()

    plt.show()
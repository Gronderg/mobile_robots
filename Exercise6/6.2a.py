import numpy as np
import matplotlib.pyplot as plt
from visualize_mobile_robot import sim_mobile_robot
import ctypes
# Constants and Settings
Ts = 0.01 # Update simulation every 10ms
t_max = 70 # total simulation duration in seconds
# Set initial state
init_state = np.array([0., 0., -np.pi/2]) # px, py, theta
IS_SHOWING_2DVISUALIZATION = True
ROBOT_RADIUS = 0.21
WHEEL_RADIUS = 0.1
MAX_ROT_SPEED = 10
# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)


# IMPLEMENTATION FOR THE CONTROLLER
#---------------------------------------------------------------------
def compute_control_input(desired_state, robot_state, next_desired_state):
    # initial numpy array for [vlin, omega]
    current_input = np.array([0., 0.]) 

    l1 = 0.08
    l2 = 0.
    
    theta = robot_state[2]
    B = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    H = B@np.array([[1., 0.], [0., l1]])@np.array([[1., -l2], [0., 1.]])

    L = np.array([l1, l2])

    # S = np.array([[0.], [0.]])
    S = robot_state[0:2] + B@L
    # Compute the control input
    u_bar = np.array([[0.], [0.]])

    k = gtg_control_input(desired_state, robot_state)
    
    u_bar = k * (desired_state[0:2] - S) + next_desired_state
    # u_bar = k * (desired_state[0:2] - S) + (next_desired_state[0:2]-desired_state[0:2])/Ts
    current_input = u_bar #np.linalg.inv(H)@u_bar
    return current_input, S

def gtg_control_input(desired_state, robot_state):
    v0 = 0.19
    beta = 5.
    error = (desired_state-robot_state)
    norm_er_pos = np.sqrt(error[0]**2 + error[1]**2)
    err = np.exp(-beta*norm_er_pos)
    k_g = v0*(1-err)/norm_er_pos

    return k_g


# MAIN SIMULATION COMPUTATION
#---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max/Ts) # Total Step for simulation
    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy() # numpy array for [px, py, theta]
    desired_state = np.array([0., 0., 0.]) # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros( (sim_iter, len(robot_state)) )
    single_integrator_his = np.zeros( (sim_iter, 2) )
    goal_history = np.zeros( (sim_iter, len(desired_state)) ) 
    input_history = np.zeros( (sim_iter, 2) ) # for [vlin, omega] vs iteration time
    right_history = np.zeros( (sim_iter, 1) )
    left_history = np.zeros( (sim_iter, 1) )

    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        # sim_visualizer = sim_mobile_robot( 'omnidirectional' ) # Omnidirectional Icon
        sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field( field_x, field_y ) # set plot area
        sim_visualizer.show_goal(desired_state)

    for it in range(sim_iter):
        current_time = it*Ts
        # record current state at time-step t
        state_history[it] = robot_state
        p_x_dot = 0.5*np.sin(0.25*current_time)
        p_y_dot = -0.5*np.cos(0.5*current_time)

        desired_state = np.array([-2*np.cos(0.25*current_time), -np.sin(0.5*current_time),
                                   np.arctan2(p_y_dot, p_x_dot)])
        goal_history[it] = desired_state

        # COMPUTE CONTROL INPUT
        #------------------------------------------------------------
        feedforward =  np.array([p_x_dot,p_y_dot])
        # feedforward = np.array([-2*np.cos(0.25*next_time), -np.sin(0.5*next_time),
        #                            np.arctan2(-0.5*np.cos(0.5*next_time), 0.5*np.sin(0.25*next_time))])
        current_input, S = compute_control_input(desired_state, robot_state, feedforward)
        #------------------------------------------------------------
        w_r = (current_input[0] + ROBOT_RADIUS*current_input[1])/WHEEL_RADIUS
        w_l = (current_input[0] - ROBOT_RADIUS*current_input[1])/WHEEL_RADIUS
        right_history[it] = w_r
        left_history[it] = w_l
        if w_r > MAX_ROT_SPEED:
            ctypes.windll.user32.MessageBoxW(0, "Right", "Warming", 1)
        if w_l > MAX_ROT_SPEED:
            ctypes.windll.user32.MessageBoxW(0, "Left", "Warming", 1)
        # record the computed input at time-step t
        input_history[it] = current_input
        single_integrator_his[it] = S.T

        if IS_SHOWING_2DVISUALIZATION: # Update Plot
            sim_visualizer.update_time_stamp( current_time )
            sim_visualizer.update_goal( desired_state )
            sim_visualizer.update_trajectory( state_history[:it+1] ) # up to the latest data
        
        #--------------------------------------------------------------------------------
        # Update new state of the robot at time-step t+1
        # using discrete-time model of UNICYCLE model
        theta = robot_state[2]
        B = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
        robot_state = robot_state + Ts*(B @ current_input) # will be used in the next iteration
        robot_state[2] = ( (robot_state[2] + np.pi) % (2*np.pi) ) - np.pi # ensure theta within [-pi pi]

        # Update desired state if we consider moving goal position
        #desired_state = desired_state + Ts*(-1)*np.ones(len(robot_state))

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, single_integrator_his, goal_history, input_history, right_history, left_history


if __name__ == '__main__':
    
    # Call main computation for robot simulation
    state_history, single_integrator_his, goal_history, input_history, right_history, left_history = simulate_control()


    #----------------------------------------------
    t = [i*Ts for i in range( round(t_max/Ts) )]
    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    ax.plot(t, input_history[:,0], label='vx [m/s]')
    ax.plot(t, input_history[:,1], label='omega [rad/s]')
    ax.set(xlabel="t [s]", ylabel="control input")
    plt.legend()
    plt.grid()

    # Plot historical data of state
    fig3 = plt.figure(3)
    ax = plt.gca()
    ax.plot(t, goal_history[:,0] - state_history[:,0], label='error x_d - x in x')
    ax.plot(t, goal_history[:,1] - state_history[:,1], label='error x_d - x in y')
    ax.set(xlabel="t [s]", ylabel="m")
    plt.legend()
    plt.grid()

    # Plot historical data of state
    fig4 = plt.figure(4)
    ax = plt.gca()
    ax.plot(t, goal_history[:,0] - single_integrator_his[:,0], label='error x_d - S in x')
    ax.plot(t, goal_history[:,1] - single_integrator_his[:,1], label='error x_d - S in y')
    ax.set(xlabel="t [s]", ylabel="m")
    plt.legend()
    plt.grid()


    # Plot historical data of state
    fig5 = plt.figure(5)
    ax = plt.gca()
    ax.plot(t, right_history[:], label='right wheel speed')
    ax.plot(t, left_history[:], label='left wheel speed')
    ax.set(xlabel="t [s]", ylabel="rad/s")
    plt.legend()
    plt.grid()
    
    # Plot historical data of state
    fig6 = plt.figure(6)
    ax = plt.gca()
    ax.plot(t, state_history[:,0], label='px [m]')
    ax.plot(t, state_history[:,1], label='py [m]')
    ax.plot(t, goal_history[:,0], ':', label='goal px [m]')
    ax.plot(t, goal_history[:,1], ':', label='goal py [m]')
    ax.set(xlabel="t [s]", ylabel="m")
    plt.legend()
    plt.grid()

    plt.show()

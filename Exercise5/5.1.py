import numpy as np
import matplotlib.pyplot as plt
from visualize_mobile_robot import sim_mobile_robot
import math

# Constants and Settings
Ts = 0.01 # Update simulation every 10ms
t_max = 15 # total simulation duration in seconds
# Set initial state
init_state = np.array([-2., 0.5, 0.]) # px, py, theta
ROBOT_RADIUS = 0.21
MAX_TRANS_VEL = 0.5
MAX_ROT_VEL = 5
d_safe = 0.75
eps = 0.01
IS_SHOWING_2DVISUALIZATION = True

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)

# IMPLEMENTATION FOR THE CONTROLLER
#---------------------------------------------------------------------
def gtg_control_input(desired_state, robot_state):
    # Feel free to adjust the input and output of the function as needed.
    # And make sure it is reflected inside the loop in simulate_control()
    mag_err = np.sqrt((desired_state[1]-robot_state[1])**2+(desired_state[0]-robot_state[0])**2)
    v0 = 5
    beta = 0.5
    err = math.exp(-beta*mag_err)
    k_g = v0*(1-err)/mag_err
    # initial numpy array for [vx, vy, omega]
    current_input = np.array([0., 0., 0.]) 
    # Compute the control input
    current_input = k_g*(desired_state-robot_state) 

    return current_input

def avo_control_input(obstacle, robot_state, obs_dis):
    # Feel free to adjust the input and output of the function as needed.
    # And make sure it is reflected inside the loop in simulate_control()
    k_o = (1/obs_dis)*(5/(obs_dis**2+eps))
    # initial numpy array for [vx, vy, omega]
    current_input = np.array([0., 0., 0.]) 
    # Compute the control input
    current_input = k_o*(robot_state-obstacle)

    return current_input


# MAIN SIMULATION COMPUTATION
#---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max/Ts) # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy() # numpy array for [px, py, theta]
    desired_state = np.array([2., -1., 0.]) # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros( (sim_iter, len(robot_state)) ) 
    goal_history = np.zeros( (sim_iter, len(desired_state)) ) 
    input_history = np.zeros( (sim_iter, 3) ) # for [vx, vy, omega] vs iteration time
    obs_history = np.zeros( (sim_iter, 1) )
    speed_history = np.zeros( (sim_iter, 1) )

    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        sim_visualizer = sim_mobile_robot( 'omnidirectional' ) # Omnidirectional Icon
        #sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field( field_x, field_y ) # set plot area
        sim_visualizer.show_goal(desired_state)
        sim_visualizer.ax.add_patch(plt.Circle((0,0),0.5,color='r'))
        sim_visualizer.ax.add_patch(plt.Circle((0,0),d_safe,color='r',fill = False))
        sim_visualizer.ax.add_patch(plt.Circle((0,0),d_safe+eps,color='g', fill=False))


    for it in range(sim_iter):
        obstacle = np.array([0., 0., 0.])
        current_time = it*Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        obs_dis = np.sqrt((robot_state[1]-obstacle[1])**2+(robot_state[0]-obstacle[0])**2)
        #err = np.arctan2((desired_state[1]-robot_state[1]),(desired_state[0]-robot_state[0]))
        # COMPUTE CONTROL INPUT
        #------------------------------------------------------------
        print(robot_state)
        if obs_dis < d_safe:
            current_input = avo_control_input(obstacle, robot_state, obs_dis)
        elif obs_dis >= d_safe + eps:
            current_input = gtg_control_input(desired_state, robot_state)
        #------------------------------------------------------------
        speed = np.sqrt(current_input[0]**2 + current_input[1]**2)
        if speed > MAX_TRANS_VEL:
            current_input[0] = current_input[0]*MAX_TRANS_VEL/speed
            current_input[1] = current_input[1]*MAX_TRANS_VEL/speed
            speed = np.sqrt(current_input[0]**2 + current_input[1]**2)

        # record the computed input at time-step t
        input_history[it] = current_input
        obs_history[it] = obs_dis
        speed_history[it] = speed

        if IS_SHOWING_2DVISUALIZATION: # Update Plot
            sim_visualizer.update_time_stamp( current_time )
            sim_visualizer.update_goal( desired_state )
            sim_visualizer.update_trajectory( state_history[:it+1] ) # up to the latest data
        
        #--------------------------------------------------------------------------------
        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        robot_state = robot_state + Ts*current_input # will be used in the next iteration
        robot_state[2] = ( (robot_state[2] + np.pi) % (2*np.pi) ) - np.pi # ensure theta within [-pi pi]

        # Update desired state if we consider moving goal position
        #desired_state = desired_state + Ts*(-1)*np.ones(len(robot_state))

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, goal_history, input_history, obs_history, speed_history


if __name__ == '__main__':
    
    # Call main computation for robot simulation
    state_history, goal_history, input_history, obs_history, speed_history = simulate_control()


    # ADDITIONAL PLOTTING
    #----------------------------------------------
    t = [i*Ts for i in range( round(t_max/Ts) )]

    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    ax.plot(t, input_history[:,0], label='vx [m/s]')
    ax.plot(t, input_history[:,1], label='vy [m/s]')
    ax.plot(t, input_history[:,2], label='omega [rad/s]')
    ax.plot(t, speed_history, label='speed of robot')
    ax.set(xlabel="t [s]", ylabel="control input")
    plt.legend()
    plt.grid()
    plt.title("time series of control input u and speed")

    fig3 = plt.figure(3)
    ax = plt.gca()
    ax.plot(t, goal_history[:,0] - state_history[:,0], label='error of x')
    ax.plot(t, goal_history[:,1] - state_history[:,1], label='error of y')
    ax.plot(t, goal_history[:,2] - state_history[:,2], label='error of omega')
    ax.set(xlabel="t [s]", ylabel="x_d - x")
    plt.legend()
    plt.grid()
    plt.title("time series of error ||x_d - x||")

    fig4 = plt.figure(4)
    ax = plt.gca()
    ax.plot(t, obs_history, label='distance from x to x_o')
    ax.set(xlabel="t [s]", ylabel="||x - x_o||")
    plt.legend()
    plt.grid()
    plt.title("time series of distance to obstacle ||x - x_o||")

    # Plot historical data of state
    fig5 = plt.figure(5)
    ax = plt.gca()
    ax.plot(t, state_history[:,0], label='px [m]')
    ax.plot(t, state_history[:,1], label='py [m]')
    ax.plot(t, state_history[:,2], label='theta [rad]')
    ax.plot(t, goal_history[:,0], ':', label='goal px [m]')
    ax.plot(t, goal_history[:,1], ':', label='goal py [m]')
    ax.plot(t, goal_history[:,2], ':', label='goal theta [rad]')
    ax.set(xlabel="t [s]", ylabel="state")
    plt.legend()
    plt.grid()
    plt.title("time series of state trajectory x vs x_d")
    plt.show()

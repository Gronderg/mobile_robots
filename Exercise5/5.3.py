import numpy as np
import matplotlib.pyplot as plt
from visualize_mobile_robot import sim_mobile_robot
import math
import cvxopt

# Constants and Settings
Ts = 0.01 # Update simulation every 10ms
t_max = 10 # total simulation duration in seconds
# Set initial state
init_state = np.array([-2., -1., 0.]) # px, py, theta
obs1 = np.array([-0.6, 0., 0.])
obs2 = np.array([0.6, 0., 0.])
#obs3 = np.array([1.05, 0.8, 0.])
ROBOT_RADIUS = 0.2
MAX_TRANS_VEL = 0.5
MAX_ROT_VEL = 5
d_safe = 0.55
d_safe2 = 0.65
eps = 0.01
# alpha = 0.5
IS_SHOWING_2DVISUALIZATION = True

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)

# IMPLEMENTATION FOR THE CONTROLLER
#---------------------------------------------------------------------
def compute_control_input(desired_state, robot_state, u_gtg_hist, it):
    # Feel free to adjust the input and output of the function as needed.
    # And make sure it is reflected inside the loop in simulate_control()

    u_gtg = gtg_control_input(desired_state, robot_state)
    speed = np.sqrt(u_gtg[0]**2 + u_gtg[1]**2)
    # if speed > 0.5:
    #     u_gtg[0] = u_gtg[0]*MAX_TRANS_VEL/speed
    #     u_gtg[1] = u_gtg[1]*MAX_TRANS_VEL/speed
    current_input, h = QP_opt(u_gtg, robot_state)
    u_gtg_hist[it] = u_gtg

    # print(func_h[0,:].shape)

    return current_input, u_gtg_hist, h

def gtg_control_input(desired_state, robot_state):
    # Feel free to adjust the input and output of the function as needed.
    # And make sure it is reflected inside the loop in simulate_control()
    mag_err = np.sqrt((desired_state[1]-robot_state[1])**2+(desired_state[0]-robot_state[0])**2)
    v0 = 3
    beta = 0.4
    err = math.exp(-beta*mag_err)
    k_g = v0*(1-err)/mag_err
    # initial numpy array for [vx, vy, omega]
    current_input = np.array([0., 0., 0.]) 
    # Compute the control input
    current_input = k_g*(desired_state-robot_state) 

    return current_input

def QP_opt(u_gtg,rb):
    # QP-based controller
    Q_mat = 2* cvxopt.matrix(np.eye(2), tc='d')
    c_mat = -2 * cvxopt.matrix(u_gtg[:2], tc='d')

    #Fill H and b based on the specification afterwards
    #row is the number of constraints
    H = cvxopt.matrix(-2*np.array([[(rb[0]-obs1[0]),(rb[1]-obs1[1])],
                                    [(rb[0]-obs2[0]),(rb[1]-obs2[1])]]), tc='d')
    h_ = np.array([[(rb[1]-obs1[1])**2+(rb[0]-obs1[0])**2 - d_safe**2],
                        [(rb[1]-obs2[1])**2+(rb[0]-obs2[0])**2 - d_safe2**2]])
    h = cvxopt.matrix(h_, tc='d')
    b = 10*h
    
    H_mat = cvxopt.matrix(H, tc='d')
    b_mat = cvxopt.matrix(b, tc='d')

    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(Q_mat, c_mat, H_mat, b_mat, verbose=False)
    current_input = np.array([sol['x'][0], sol['x'][1],0])
    return current_input, h


# MAIN SIMULATION COMPUTATION
#---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max/Ts) # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy() # numpy array for [px, py, theta]
    desired_state = np.array([2., 1., 0.]) # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros( (sim_iter, len(robot_state)) ) 
    goal_history = np.zeros( (sim_iter, len(desired_state)) ) 
    input_history = np.zeros( (sim_iter, 3) ) # for [vx, vy, omega] vs iteration time
    u_gtg_hist = np.zeros( (sim_iter, 3) )
    func_h = np.zeros( (sim_iter, 3) )

    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        sim_visualizer = sim_mobile_robot( 'omnidirectional' ) # Omnidirectional Icon
        #sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field( field_x, field_y ) # set plot area
        sim_visualizer.show_goal(desired_state)

        sim_visualizer.ax.add_patch(plt.Circle((-0.6,0),0.3,color='r'))
        sim_visualizer.ax.add_patch(plt.Circle((-0.6,0),d_safe,color='r',fill = False))
        #sim_visualizer.ax.add_patch(plt.Circle((-0.,0),d_safe+eps,color='g', fill=False))

        sim_visualizer.ax.add_patch(plt.Circle((0.6,0.),0.4,color='r'))
        sim_visualizer.ax.add_patch(plt.Circle((0.6,0.),d_safe2,color='r',fill = False))
        #sim_visualizer.ax.add_patch(plt.Circle((0.6,0.),d_safe2+eps,color='g', fill=False))

        # sim_visualizer.ax.add_patch(plt.Circle((1.05,0.8),0.3,color='r'))
        # sim_visualizer.ax.add_patch(plt.Circle((1.05,0.8),d_safe,color='r',fill = False))
        # sim_visualizer.ax.add_patch(plt.Circle((1.05,0.8),d_safe+eps,color='g', fill=False))

    for it in range(sim_iter):
        current_time = it*Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state


        # COMPUTE CONTROL INPUT
        #------------------------------------------------------------
        current_input, u_gtg_hist, h = compute_control_input(desired_state, robot_state,
                                                                   u_gtg_hist, it)
        #------------------------------------------------------------
        #func_h[it] = np.transpose(h)
        # speed = np.sqrt(current_input[0]**2 + current_input[1]**2)
        # if speed > 0.5:
        #     current_input[0] = current_input[0]*0.5/speed
        #     current_input[1] = current_input[1]*0.5/speed
            # speed = np.sqrt(current_input[0]**2 + current_input[1]**2)

        #if (current_input[0,:]**2 + current_input[1,:]**2) > 0.5: 
            
        # record the computed input at time-step t
        input_history[it] = current_input

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
    return state_history, goal_history, input_history, u_gtg_hist

if __name__ == '__main__':
    
    # Call main computation for robot simulation
    state_history, goal_history, input_history, u_gtg_hist = simulate_control()


    # ADDITIONAL PLOTTING
    #----------------------------------------------
    t = [i*Ts for i in range( round(t_max/Ts) )]

    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    ax.plot(t, input_history[:,0], label='vx [m/s]')
    ax.plot(t, input_history[:,1], label='vy [m/s]')
    ax.plot(t, u_gtg_hist[:,0], label='vx_gtg [m/s]')
    ax.plot(t, u_gtg_hist[:,1], label='vy_gtg [m/s]')
    ax.set(xlabel="t [s]", ylabel="control inputs")
    plt.legend()
    plt.title("Time series comparison of control input u_gtg and u")
    plt.grid()

    # # Plot historical data of state
    # fig3 = plt.figure(3)
    # ax = plt.gca()
    # ax.plot(t, func_h[:,0], label='h_o1')
    # ax.plot(t, func_h[:,1], label='h_o2')
    # ax.plot(t, func_h[:,2], label='h_o3')
    # ax.set(xlabel="t [s]", ylabel="function h")
    # plt.legend()
    # plt.title("Time series comparison of function h_o1, ho_2, and ho_3")
    # plt.grid()

    plt.show()

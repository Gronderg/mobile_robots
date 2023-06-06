import numpy as np
import matplotlib.pyplot as plt
from visualize_mobile_robot import sim_mobile_robot
from ex6p3_obstacles import dict_obst_vertices
from detect_obstacle import DetectObstacle
import ctypes
 
# Constants and Settings
Ts = 0.01 # Update simulation every 10ms
t_max = 140 # total simulation duration in seconds
# Set initial state
init_state = np.array([-4., -3.5, 0.]) # px, py, theta
IS_SHOWING_2DVISUALIZATION = True
 


# Define Field size for plotting (should be in tuple)
field_x = (-5, 5)
field_y = (-4, 4)

# Define Obstacles 
obst_vertices = np.array( [ [-1., -1.2], [1., -1.2], [1., 1.2], [-1., 1.2], \
        [-1., 0.8], [0, 0.8], [0.5, 0.5], [0.5, -0.5], [0., -0.8], [-1., -0.8], [-1., -1.2]]) 

# Define sensor's sensing range and resolution
sensing_range = 1. # in meter
sensor_resolution = np.pi/8 # angle between sensor data in radian


# IMPLEMENTATION FOR THE CONTROLLER
#---------------------------------------------------------------------
def compute_sensor_endpoint(robot_state, sensors_dist):
    # assuming sensor position is in the robot's center
    sens_N = round(2*np.pi/sensor_resolution)
    sensors_theta = [i*2*np.pi/sens_N for i in range(sens_N)]
    obst_points = np.zeros((3,sens_N))

    R_WB = np.array([ [np.cos(robot_state[2]), -np.sin(robot_state[2]), robot_state[0] ], \
        [np.sin(robot_state[2]),  np.cos(robot_state[2]), robot_state[1] ], [0, 0, 1] ])
    for i in range(sens_N):
        R_BS = np.array([ [np.cos(sensors_theta[i]), -np.sin(sensors_theta[i]), 0 ], \
            [np.sin(sensors_theta[i]),  np.cos(sensors_theta[i]), 0 ], [0, 0, 1] ])
        temp = R_WB @ R_BS @ np.array([sensors_dist[i], 0, 1])
        obst_points[:,i] = temp

    return obst_points[:2,:]

ROBOT_RADIUS = 0.21
eps = 0.02
d_safe = 0.3
WHEEL_RADIUS = 0.1
MAX_ROT_SPEED = 10
l1 = 0.06
l2 = 0.

#gain for u_gtg
v0 = 0.2
beta = 1.5

#gain for u_avo
c = 0.07
#gain for wall following
alpha1 = 0.4
alpha2 = 0.25
k_wf = 0.4


def compute_control_input(desired_state, robot_state,
                           distance_reading, obst_points, control_state, wf_switch_dist):
    # initial numpy array for [vx, vy, omega]
    current_input = np.array([0., 0.])
    condi_1 = False
    condi_2 = False
    condi_3 = False
    condi_4 = False
    condi_5 = False
    condi_6 = False

    theta = robot_state[2]
    B = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    L = np.array([l1, l2])
    S_uni = robot_state[0:2] + B@L

    error = (desired_state[:2]-S_uni)
    norm_er_pos = np.sqrt(error[0]**2 + error[1]**2)
    u_gtg = gtg_control_input(desired_state, S_uni, norm_er_pos)
    print("u_gtg = ", end="")
    print(u_gtg)

    # except go to goal, all vector computations are in 2-element (x,y)
    # Compute u_avo with equal weighting (but divided by n)
    n_obs = 0
    u_avo = np.zeros(2)
    for i, dist in enumerate(distance_reading):
        if dist < 0.99*sensing_range:
            n_obs += 1
            # Compute control input for avoidance at each detected point            
            u_avo_i = avo_control_input(obst_points[:,i], S_uni, dist)
            u_avo = u_avo + u_avo_i
    print("u_avo = ", end="")
    print(u_avo)

    # if n_obs > 0: u_avo = u_avo / n_obs
    # 
    # Compute u_wf
    x_o = None
    dist_to_obs = sensing_range
    u_wfcw = np.zeros(2)
    u_wfcc = np.zeros(2)

    if n_obs > 0:
        # find order of index from min to max
        idx_min2max = np.argsort(distance_reading)
        # if False
        if n_obs > 1:
            # determine the "left" and "right"
            sorted_min2 = sorted(idx_min2max[:2])
            # in general the smaller index is the "right" side --> x_o2
            idx_xo1 = sorted_min2[1] #left
            idx_xo2 = sorted_min2[0] #right
            if (sorted_min2[1] - sorted_min2[0]) > 4:
                idx_xo1 = sorted_min2[0] #left
                idx_xo2 = sorted_min2[1] #right

            x_o = np.array([0.5*(obst_points[0,idx_xo1]+obst_points[0,idx_xo2]),
                            0.5*(obst_points[1,idx_xo1]+obst_points[1,idx_xo2])])
            
            #compute wf_cc
            u_wft_cc = obst_points[:,idx_xo2] - obst_points[:, idx_xo1]
            norm_u_wft_cc = np.sqrt(u_wft_cc[0]*u_wft_cc[0] + u_wft_cc[1]*u_wft_cc[1])
            ubar_wft_cc = u_wft_cc/norm_u_wft_cc

            disx_xo1_x = obst_points[:, idx_xo1] - S_uni
            u_pt_cc = np.inner(disx_xo1_x, ubar_wft_cc)*ubar_wft_cc
            u_wfp_cc = disx_xo1_x - u_pt_cc
            norm_u_wfp_cc = np.sqrt(u_wfp_cc[0]*u_wfp_cc[0] + u_wfp_cc[1]*u_wfp_cc[1])
            uhat_wfp_cc = u_wfp_cc - (d_safe/norm_u_wfp_cc)*u_wfp_cc

            u_wfcc = alpha1*uhat_wfp_cc + alpha2*ubar_wft_cc

            #compute wf_cw
            u_wft_cw = obst_points[:,idx_xo1] - obst_points[:, idx_xo2]
            norm_u_wft_cw = np.sqrt(u_wft_cw[0]*u_wft_cw[0] + u_wft_cw[1]*u_wft_cw[1])
            ubar_wft_cw = u_wft_cw/norm_u_wft_cw

            disx_xo2_x = obst_points[:, idx_xo2] - S_uni
            u_pt_cw = np.inner(disx_xo2_x, ubar_wft_cw)*ubar_wft_cw
            u_wfp_cw = disx_xo2_x - u_pt_cw
            norm_u_wfp_cw = np.sqrt(u_wfp_cw[0]*u_wfp_cw[0] + u_wfp_cw[1]*u_wfp_cw[1])
            uhat_wfp_cw = u_wfp_cw - (d_safe/norm_u_wfp_cw)*u_wfp_cw

            u_wfcw = alpha1*uhat_wfp_cw + alpha2*ubar_wft_cw

        else: #only one sensor is detected
            x_o = np.array([obst_points[0,idx_min2max[0]], obst_points[1,idx_min2max[0]]])
            # recompute u_avo
            u_wfcw = k_wf * np.array([[0, 1], [-1, 0]])@np.transpose(u_avo)
            u_wfcc = k_wf * np.array([[0, -1], [1, 0]])@np.transpose(u_avo)

        error_obst = (S_uni - x_o)
        dist_to_obs = np.sqrt(error_obst[0]*error_obst[0]+error_obst[1]*error_obst[1])
    
    print("u_wfcc = ", end="")
    print(u_wfcc)
    print("u_wfcw = ", end="")
    print(u_wfcw)
    # check switching condition
    if (d_safe - eps) <= dist_to_obs and dist_to_obs <= (d_safe + eps):
        condi_1 = True
        print("condi_1")
    if u_gtg@np.transpose(u_wfcc) > 0:
        condi_3 = True
        print("condi_3")
    if u_gtg@np.transpose(u_wfcw) > 0:
        condi_2 = True
        print("condi_2")
    if u_avo@np.transpose(u_gtg) > 0:
        condi_4 = True
        print("condi_4")
    if norm_er_pos < wf_switch_dist:
        condi_5 = True
        print("condi_5")
    if dist_to_obs < (d_safe - eps):
        condi_6 = True
        print("condi_6")

    #---------------------------------------------------------------
    if condi_1 and condi_2 and control_state == "GOTOTGOAL_STATE":
        wf_switch_dist = norm_er_pos
        control_state = "WF_CW_STATE"
    elif condi_1 and condi_2 and control_state == "AVO_STATE":
        control_state = "WF_CW_STATE"
    elif condi_6 and control_state == "WF_CC_STATE":
        control_state = "AVO_STATE"
    elif condi_6 and control_state == "WF_CC_STATE":
        control_state = "AVO_STATE"
    elif condi_1 and condi_3 and control_state == "GOTOTGOAL_STATE":
        wf_switch_dist = norm_er_pos
        control_state = "WF_CC_STATE"
    elif condi_1 and condi_3 and control_state == "AVO_STATE":
        control_state = "WF_CC_STATE"
    elif condi_4 and condi_5 and control_state == "WF_CW_STATE":
        control_state = "GOTOTGOAL_STATE"
    elif condi_4 and condi_5 and control_state == "WF_CC_STATE":
        control_state = "GOTOTGOAL_STATE"

    H_matrix = B@np.array([[1, 0], [0, l1]])@np.array([[1, -l2], [0, 1]])
    #Compute the control input to goal
    if control_state == "GOTOTGOAL_STATE": current_input = np.linalg.inv(H_matrix)@u_gtg
    elif control_state == "AVO_STATE": current_input = np.linalg.inv(H_matrix)@u_avo
    elif control_state == "WF_CW_STATE": current_input = np.linalg.inv(H_matrix)@u_wfcw
    elif control_state == "WF_CC_STATE": current_input = np.linalg.inv(H_matrix)@u_wfcc


    return current_input, u_gtg, u_avo, control_state, wf_switch_dist, S_uni

def gtg_control_input(desired_state, S_uni,norm_er_pos):
    err = np.exp(-beta*norm_er_pos)
    k_g = v0*(1-err)/norm_er_pos
    # initial numpy array for [vx, vy, omega]
    current_input = np.array([0., 0.]) 
    # Compute the control input
    current_input = k_g*(desired_state[:2]-S_uni)

    return current_input

def avo_control_input(x_o, robot_state, robot_2_obst):
    k_o = (1/robot_2_obst)*(c/(robot_2_obst**2+eps))
    # initial numpy array for [vx, vy, omega]
    u_avo = np.array([0., 0.])

    u_avo = k_o*(robot_state-x_o)
    return u_avo

# MAIN SIMULATION COMPUTATION
#---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max/Ts) # Total Step for simulation
    reached_goal_counter = 0
    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy() # numpy array for [px, py, theta]
    goal_1 = np.array([4., 0., 0.])
    goal_2 = np.array([-0.5, 3.7, 0.])
    goal_3 = np.array([-4., -3.5, 0.])
    desired_state =  goal_1

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros( (sim_iter, len(robot_state)) ) 
    goal_history = np.zeros( (sim_iter, len(desired_state)) )
    single_integrator_his = np.zeros( (sim_iter, 2) )
    input_history = np.zeros( (sim_iter, 2) ) # for [vx, vy, omega] vs iteration time
    right_history = np.zeros( (sim_iter, 1) )
    left_history = np.zeros( (sim_iter, 1) )

    # Initiate the Obstacle Detection
    range_sensor = DetectObstacle( sensing_range, sensor_resolution)
    for key, value in dict_obst_vertices.items():
        range_sensor.register_obstacle_bounded( value )
    control_state = "GOTOTGOAL_STATE"
    wf_switch_dist = 0

    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        sim_visualizer = sim_mobile_robot( 'unicycle' ) # Omnidirectional Icon
        #sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field( field_x, field_y ) # set plot area
        sim_visualizer.show_goal(desired_state)

        # Display the obstacle
        for key, value in dict_obst_vertices.items():
            sim_visualizer.ax.plot( value[:,0], value[:,1], '--r' )
        
        # get sensor reading
        # Index 0 is in front of the robot. 
        # Index 1 is the reading for 'sensor_resolution' away (counter-clockwise) from 0, and so on for later index
        distance_reading = range_sensor.get_sensing_data( robot_state[0], robot_state[1], robot_state[2])
        # compute and plot sensor reading endpoint
        obst_points = compute_sensor_endpoint(robot_state, distance_reading)
        pl_sens, = sim_visualizer.ax.plot(obst_points[0], obst_points[1], '.') #, marker='X')
        pl_txt = [sim_visualizer.ax.text(obst_points[0,i], obst_points[1,i], str(i)) for i in range(len(distance_reading))]

        pl_q_gtg = plt.quiver(robot_state[0], robot_state[1], 0, 0, scale_units='xy', scale=1,color='k')
        pl_q_avo = plt.quiver(robot_state[0], robot_state[1], 0, 0, scale_units='xy', scale=1,color='r')
        #pl_q_wf_cc = plt.quiver(robot_state[0], robot_state[1], 0, 0, scale_units='xy', scale=1,color='b')

    for it in range(sim_iter):
        dist_2_goal = np.sqrt((robot_state[0] - desired_state[0])**2 + (robot_state[1] - desired_state[1])**2)
        if dist_2_goal <= 0.2 and reached_goal_counter == 0:
            reached_goal_counter = reached_goal_counter + 1
            desired_state = goal_2
        elif dist_2_goal <= 0.2 and reached_goal_counter == 1:
            desired_state = goal_3
        current_time = it*Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # Get information from sensors
        distance_reading = range_sensor.get_sensing_data( robot_state[0], robot_state[1], robot_state[2])
        obst_points = compute_sensor_endpoint(robot_state, distance_reading)

        # COMPUTE CONTROL INPUT
        #------------------------------------------------------------
        current_input, u_gtg, u_avo, control_state, wf_switch_dist, S_uni = compute_control_input(desired_state, robot_state,
                           distance_reading, obst_points, control_state, wf_switch_dist)
        
        #------------------------------------------------------------
        w_r = (current_input[0] + ROBOT_RADIUS*current_input[1])/WHEEL_RADIUS
        w_l = (current_input[0] - ROBOT_RADIUS*current_input[1])/WHEEL_RADIUS
        right_history[it] = w_r
        left_history[it] = w_l
        if w_r > MAX_ROT_SPEED:
            ctypes.windll.user32.MessageBoxW(0, "Right", control_state, 1)
        if w_l > MAX_ROT_SPEED:
            ctypes.windll.user32.MessageBoxW(0, "Left", control_state, 1)
        # record the computed input at time-step t
        print(control_state)
        print("-------------------------------------")
        input_history[it] = current_input
        single_integrator_his[it] = S_uni


        if IS_SHOWING_2DVISUALIZATION: # Update Plot
            sim_visualizer.update_time_stamp( current_time )
            sim_visualizer.update_goal( desired_state )
            sim_visualizer.update_trajectory( state_history[:it+1] ) # up to the latest data
            # update sensor visualization
            pl_sens.set_data(obst_points[0], obst_points[1])
            for i in range(len(distance_reading)): pl_txt[i].set_position((obst_points[0,i], obst_points[1,i]))

            #updating the position
            pl_q_gtg.set_offsets(robot_state[:2])
            #updating the orientation
            pl_q_gtg.set_UVC(u_gtg[0], u_gtg[1])

            pl_q_avo.set_offsets(robot_state[:2])
            pl_q_avo.set_UVC(u_avo[0], u_avo[1])

        #--------------------------------------------------------------------------------
        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        theta = robot_state[2]
        B = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
        robot_state = robot_state + Ts*(B @ current_input) # will be used in the next iteration
        robot_state[2] = ( (robot_state[2] + np.pi) % (2*np.pi) ) - np.pi # ensure theta within [-pi pi]

        # Update desired state if we consider moving goal position
        #desired_state = desired_state + Ts*(-1)*np.ones(len(robot_state))

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, goal_history, input_history, right_history, left_history, single_integrator_his


if __name__ == '__main__':
    
    # Call main computation for robot simulation
    state_history, goal_history, input_history, right_history, left_history, single_integrator_his = simulate_control()


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
    ax.plot(t, goal_history[:,0] - state_history[:,0], label='x_d_x - p_x')
    ax.plot(t, goal_history[:,1] - state_history[:,1], label='x_d_y - p_y')
    ax.set(xlabel="t [s]", ylabel="m")
    plt.legend()
    plt.grid()

    # Plot historical data of state
    fig4 = plt.figure(4)
    ax = plt.gca()
    ax.plot(t, goal_history[:,0] - single_integrator_his[:,0], label='x_d_x - S_x')
    ax.plot(t, goal_history[:,1] - single_integrator_his[:,1], label='x_d_y - S_y')
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

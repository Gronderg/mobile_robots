import numpy as np
import matplotlib.pyplot as plt
from visualize_mobile_robot import sim_mobile_robot
from detect_obstacle import DetectObstacle
import math

# Constants and Settings
Ts = 0.01 # Update simulation every 10ms
t_max = 30 # total simulation duration in seconds
# Set initial state
init_state = np.array([-2., 1., 0.]) # px, py, theta
IS_SHOWING_2DVISUALIZATION = True
ROBOT_RADIUS = 0.21
MAX_TRANS_VEL = 0.5
MAX_ROT_VEL = 5
eps = 0.01
d_safe = 0.3

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)

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


def compute_control_input(desired_state, robot_state,
                           distance_reading, obst_points, control_state, radius_tx):
    print("robot_state: ", end = "")
    print(robot_state)
    # initial numpy array for [vx, vy, omega]
    current_input = np.array([0., 0., 0.])
    condi_1 = False
    condi_2 = False
    condi_3 = False
    condi_4 = False
    condi_5 = False
    condi_6 = False
    
    mini = min(distance_reading)
    id = np.where(distance_reading==mini)[0][0]
    x_o_min = find_x_o(obst_points, id)
    print("x_o_min: ", end = "")
    print(x_o_min)
    # distance from robot to obst
    robot_2_obst = np.sqrt((robot_state[0] - x_o_min[0])**2 + (robot_state[1] - x_o_min[1])**2)  
    # distance from robot to goal
    robot_2_goal = np.sqrt((robot_state[0] - desired_state[0])**2 + (robot_state[1] - desired_state[1])**2)
    u_gtg = gtg_control_input(desired_state, robot_state, robot_2_goal)
    u_avo = avo_control_input(x_o_min, robot_state, robot_2_obst)
    print("u_gtg: ", end = "")
    print(u_gtg)
    print("u_avo: ", end = "")
    print(u_avo)

    v_wf_c = np.matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])@np.transpose(u_avo)
    v_wf_cc = np.matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])@np.transpose(u_avo)
    
    
    if (d_safe - eps) <= robot_2_obst and robot_2_obst <= (d_safe + eps):
        condi_1 = True
        print("condi_1")
    if u_gtg@np.transpose(v_wf_cc) > 0:
        condi_3 = True
        print("condi_3")
    if u_gtg@np.transpose(v_wf_c) > 0:
        condi_2 = True
        print("condi_2")
    if u_avo@np.transpose(u_gtg) > 0:
        condi_4 = True
        print("condi_4")
    if robot_2_goal < radius_tx:
        condi_5 = True
        print("condi_5")
    if robot_2_goal < (d_safe - eps):
        condi_6 = True
        print("condi_6")

    #---------------------------------------------------------------
    if condi_1 and condi_2 and control_state == "gtg":
        radius_tx = np.sqrt((robot_state[0] - desired_state[0])**2 +
                             (robot_state[1] - desired_state[1])**2)
        control_state = "wf_c"
    elif condi_1 and condi_2 and control_state == "avo":
        control_state = "wf_c"
    elif condi_6 and control_state == "wf_c":
        control_state = "avo"
    elif condi_6 and control_state == "wf_cc":
        control_state = "avo"
    elif condi_1 and condi_3 and control_state == "gtg":
        radius_tx = np.sqrt((robot_state[0] - desired_state[0])**2 +
                             (robot_state[1] - desired_state[1])**2)
        control_state = "wf_cc"
    elif condi_1 and condi_3 and control_state == "avo":
        control_state = "wf_cc"
    elif condi_4 and condi_5 and control_state == "wf_c":
        control_state = "gtg"
    elif condi_4 and condi_5 and control_state == "wf_cc":
        control_state = "gtg"
    print(control_state)

    #--------------------------------------------------------------
    # take 2 sensors near
    x_o_1 = np.array([0., 0., 0.])
    x_o_2 = np.array([0., 0., 0.])
    if control_state == "gtg":
        current_input = u_gtg
    elif control_state == "avo":
        current_input = u_avo
    elif control_state == "wf_c":
        x_o_1, x_o_2 = nearest_sensor_c(distance_reading, obst_points)
        current_input = wall_following_clockwise(robot_state, x_o_1, x_o_2)
    elif control_state == "wf_cc":
        x_o_1, x_o_2 = nearest_sensor_cc(distance_reading, obst_points)
        current_input = wall_following_cc(robot_state, x_o_1, x_o_2)

    print("u control input: ", end = "")
    print(current_input)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    return current_input, u_gtg, control_state, radius_tx, mini

def nearest_sensor_c(distance_reading, obst_points):
    # take 2 sensors near
    x_o_1 = np.array([0., 0., 0.])
    x_o_2 = np.array([0., 0., 0.])
    new = np.sort(distance_reading)
    A = new[0]
    idx1 = np.where(distance_reading==A)[0][0]
    idx2 = 16
    idx3 = 16
    if A < 0.99:
        if idx1 == 15 and distance_reading[0] != 1:
            idx2 = 0
        elif distance_reading[idx1 + 1] != 1:
            idx2 = idx1 + 1

        if distance_reading[idx1 - 1] != 1:
            idx3 = idx1 - 1
    if idx2 != 16 and idx3 != 16:
        x3 = find_x_o(obst_points, idx3)
        x2 = find_x_o(obst_points, idx2)
        x1 = find_x_o(obst_points, idx1)
        if x1[0] == x2[0] == x3[0]:
            x_o_2 = find_x_o(obst_points, idx2)
            x_o_1 = find_x_o(obst_points, idx3)
        elif x1[1] == x2[1] == x3[1] and x1[0] > x3[0]:
            x_o_2 = find_x_o(obst_points, idx2)
            x_o_1 = find_x_o(obst_points, idx3)
        elif x1[1] == x2[1] == x3[1] and x1[0] < x3[0]:
            x_o_2 = find_x_o(obst_points, idx2)
            x_o_1 = find_x_o(obst_points, idx3)
        elif (x1[0] == x3[0]) or (x1[1] == x3[1]):
            x_o_2 = find_x_o(obst_points, idx1)
            x_o_1 = find_x_o(obst_points, idx3)
        elif (x1[0] == x2[0]) or (x1[1] == x2[1]):
            x_o_2 = find_x_o(obst_points, idx2)
            x_o_1 = find_x_o(obst_points, idx1)
        else:
            x32 = x3 - x2
            if x32[1] > 0:
                x_o_2 = find_x_o(obst_points, idx3)
                x_o_1 = find_x_o(obst_points, idx2)
            else:
                x_o_2 = find_x_o(obst_points, idx2)
                x_o_1 = find_x_o(obst_points, idx3)

    elif idx2 != 16:
        x1 = find_x_o(obst_points, idx1)
        x2 = find_x_o(obst_points, idx2)
        if x1[0] == x2[0]:
            x_o_2 = find_x_o(obst_points, idx2)
            x_o_1 = find_x_o(obst_points, idx1)
        elif x1[0] < x2[0] and x1[1] == x2[1]:
            x_o_2 = find_x_o(obst_points, idx2)
            x_o_1 = find_x_o(obst_points, idx1)
        elif x1[0] > x2[0] and x1[1] == x2[1]:
            x_o_2 = find_x_o(obst_points, idx2)
            x_o_1 = find_x_o(obst_points, idx1)
        else:
            x12 = x1 - x2
            if x12[1] > 0:
                x_o_2 = find_x_o(obst_points, idx1)
                x_o_1 = find_x_o(obst_points, idx2)
            else:
                x_o_2 = find_x_o(obst_points, idx2)
                x_o_1 = find_x_o(obst_points, idx1)
        
    elif idx3 != 16:
        x11 = find_x_o(obst_points, idx1)
        x3 = find_x_o(obst_points, idx3)
        if x11[0] == x3[0]:
            x_o_2 = find_x_o(obst_points, idx1)
            x_o_1 = find_x_o(obst_points, idx3)
        elif x11[0] > x3[0] and x11[1] == x3[1]:
            x_o_2 = find_x_o(obst_points, idx1)
            x_o_1 = find_x_o(obst_points, idx3)
        elif x11[0] < x3[0] and x11[1] == x3[1]:
            x_o_2 = find_x_o(obst_points, idx1)
            x_o_1 = find_x_o(obst_points, idx3)
        else:
            x13 = x11 - x3
            if x13[1] > 0:
                x_o_2 = find_x_o(obst_points, idx1)
                x_o_1 = find_x_o(obst_points, idx3)
            else:
                x_o_2 = find_x_o(obst_points, idx3)
                x_o_1 = find_x_o(obst_points, idx1)
    
    return x_o_1, x_o_2

def nearest_sensor_cc(distance_reading, obst_points):
    # take 2 sensors near
    x_o_1 = np.array([0., 0., 0.])
    x_o_2 = np.array([0., 0., 0.])
    new = np.sort(distance_reading)
    A = new[0]
    idx1 = np.where(distance_reading==A)[0][0]
    x_o_1 = find_x_o(obst_points, idx1)
    idx2 = 16
    idx3 = 16
    if A < 0.99:
        if idx1 == 15 and distance_reading[0] != 1:
            idx2 = 0
        elif idx1 == 15 and distance_reading[0] == 1:
            idx2 = 16
        elif distance_reading[idx1 + 1] != 1:
            idx2 = idx1 + 1

        if distance_reading[idx1 - 1] != 1:
            idx3 = idx1 - 1
    if idx2 != 16 and idx3 != 16:
        x3 = find_x_o(obst_points, idx3)
        x2 = find_x_o(obst_points, idx2)
        x1 = find_x_o(obst_points, idx1)
        if x1[0] == x2[0] == x3[0]:
            x_o_2 = find_x_o(obst_points, idx3)
            x_o_1 = find_x_o(obst_points, idx2)
        elif x1[1] == x2[1] == x3[1] and x1[0] > x3[0]:
            x_o_2 = find_x_o(obst_points, idx3)
            x_o_1 = find_x_o(obst_points, idx2)
        elif x1[1] == x2[1] == x3[1] and x1[0] < x3[0]:
            x_o_2 = find_x_o(obst_points, idx3)
            x_o_1 = find_x_o(obst_points, idx2)
        elif (x1[0] == x3[0]) or (x1[1] == x3[1]):
            x_o_2 = find_x_o(obst_points, idx3)
            x_o_1 = find_x_o(obst_points, idx1)
        elif (x1[0] == x2[0]) or (x1[1] == x2[1]):
            x_o_2 = find_x_o(obst_points, idx1)
            x_o_1 = find_x_o(obst_points, idx2)
        else:
            x32 = x3 - x2
            if x32[1] < 0:
                x_o_2 = find_x_o(obst_points, idx3)
                x_o_1 = find_x_o(obst_points, idx2)
            else:
                x_o_2 = find_x_o(obst_points, idx2)
                x_o_1 = find_x_o(obst_points, idx3)

    elif idx2 != 16:
        x1 = find_x_o(obst_points, idx1)
        x2 = find_x_o(obst_points, idx2)
        if (x1[0] == x2[0]):
            x_o_2 = find_x_o(obst_points, idx1)
            x_o_1 = find_x_o(obst_points, idx2)
        elif (x1[0] < x2[0]) and (x1[1] == x2[1]):
            x_o_2 = find_x_o(obst_points, idx1)
            x_o_1 = find_x_o(obst_points, idx2)
        elif (x1[0] > x2[0]) and (x1[1] == x2[1]):
            x_o_2 = find_x_o(obst_points, idx1)
            x_o_1 = find_x_o(obst_points, idx2)
        else:
            x_o_2 = find_x_o(obst_points, idx1)
            x_o_1 = find_x_o(obst_points, idx2)

        
    elif idx3 != 16:
        x1 = find_x_o(obst_points, idx1)
        x3 = find_x_o(obst_points, idx3)
        if (x1[0] == x3[0]) :
            x_o_2 = find_x_o(obst_points, idx3)
            x_o_1 = find_x_o(obst_points, idx1)
        elif (x1[0] > x3[0]) and (x1[1] == x3[1]):
            x_o_2 = find_x_o(obst_points, idx3)
            x_o_1 = find_x_o(obst_points, idx1)
        elif (x1[0] < x3[0]) and (x1[1] == x3[1]):
            x_o_2 = find_x_o(obst_points, idx3)
            x_o_1 = find_x_o(obst_points, idx1)
        else:

            x_o_2 = find_x_o(obst_points, idx3)
            x_o_1 = find_x_o(obst_points, idx1)
    else:
        x_o_2 = find_x_o(obst_points, idx1-1)
    
    return x_o_1, x_o_2

def find_x_o(obst_points, id):
    x_o = np.array([0., 0., 0.])
    x_o[0] = obst_points[0,id]
    x_o[1] = obst_points[1,id]
    return x_o

def gtg_control_input(desired_state, robot_state, robot_2_goal):
    v0 = 3
    beta = 0.4
    err = math.exp(-beta*robot_2_goal)
    k_g = v0*(1-err)/robot_2_goal
    # initial numpy array for [vx, vy, omega]
    current_input = np.array([0., 0., 0.]) 
    # Compute the control input
    current_input = k_g*(desired_state-robot_state)

    return current_input

def avo_control_input(x_o_min, robot_state, robot_2_obst):
    c = 5
    k_o = (1/robot_2_obst)*(c/(robot_2_obst**2+eps))
    # initial numpy array for [vx, vy, omega]
    u_avo = np.array([0., 0., 0.])

    u_avo = k_o*(robot_state-x_o_min)
    return u_avo

def wall_following_cc(robot_state, x_o_1, x_o_2):
    v_tan = x_o_2 - x_o_1
    dist_x_ = np.sqrt((v_tan[0])**2 + (v_tan[1])**2)
    u_wf_tan = v_tan/dist_x_
    v_perp = (x_o_1 - robot_state) - ((x_o_1 - robot_state)*u_wf_tan)*u_wf_tan
    dist_v_perp = np.sqrt((v_perp[0])**2 + (v_perp[1])**2)
    u_wf_perp = v_perp - (d_safe/dist_v_perp)*v_perp
    u_wf_cc = u_wf_perp + u_wf_tan
    return u_wf_cc

def wall_following_clockwise(robot_state, x_o_1, x_o_2):
    v_tan = x_o_1 - x_o_2
    u_wf_tan = v_tan/np.sqrt((v_tan[0])**2 + (v_tan[1])**2)

    v_perp = (x_o_1 - robot_state) - ((x_o_1 - robot_state)*u_wf_tan)*u_wf_tan
    dist_v_perp = np.sqrt((v_perp[0])**2 + (v_perp[1])**2)
    u_wf_perp = v_perp - (d_safe/dist_v_perp)*v_perp
    u_wf_clockwise = u_wf_perp + u_wf_tan

    return u_wf_clockwise
    


# MAIN SIMULATION COMPUTATION
#---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max/Ts) # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy() # numpy array for [px, py, theta]
    desired_state = np.array([2., 0., 0.]) # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros( (sim_iter, len(robot_state)) ) 
    goal_history = np.zeros( (sim_iter, len(desired_state)) ) 
    input_history = np.zeros( (sim_iter, 3) ) # for [vx, vy, omega] vs iteration time
    speed_history = np.zeros( (sim_iter, 1) )
    min_sensor_history = np.zeros( (sim_iter, 1) )

    # Initiate the Obstacle Detection
    range_sensor = DetectObstacle( sensing_range, sensor_resolution)
    range_sensor.register_obstacle_bounded( obst_vertices )
    control_state = "gtg"
    radius_tx = 0

    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        sim_visualizer = sim_mobile_robot( 'omnidirectional' ) # Omnidirectional Icon
        #sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field( field_x, field_y ) # set plot area
        sim_visualizer.show_goal(desired_state)

        # Display the obstacle
        sim_visualizer.ax.plot( obst_vertices[:,0], obst_vertices[:,1], '--r' )
        
        # get sensor reading
        # Index 0 is in front of the robot. 
        # Index 1 is the reading for 'sensor_resolution' away (counter-clockwise) from 0, and so on for later index
        distance_reading = range_sensor.get_sensing_data( robot_state[0], robot_state[1], robot_state[2])
        # compute and plot sensor reading endpoint
        obst_points = compute_sensor_endpoint(robot_state, distance_reading)
        pl_sens, = sim_visualizer.ax.plot(obst_points[0], obst_points[1], '.') #, marker='X')
        pl_txt = [sim_visualizer.ax.text(obst_points[0,i], obst_points[1,i], str(i)) for i in range(len(distance_reading))]

        pl_q_gtg = plt.quiver(robot_state[0], robot_state[1], 0, 0, scale_units='xy', scale=1,color='k')

    for it in range(sim_iter):
        current_time = it*Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # Get information from sensors
        distance_reading = range_sensor.get_sensing_data( robot_state[0], robot_state[1], robot_state[2])
        obst_points = compute_sensor_endpoint(robot_state, distance_reading)

        # COMPUTE CONTROL INPUT
        #------------------------------------------------------------
        current_input, u_gtg, control_state, radius_tx, min_sensor = compute_control_input(desired_state, robot_state,
                                                      distance_reading, obst_points, control_state, radius_tx)
        #------------------------------------------------------------
        speed = np.sqrt(current_input[0]**2 + current_input[1]**2)
        if speed > MAX_TRANS_VEL:
            current_input[0] = current_input[0]*MAX_TRANS_VEL/speed
            current_input[1] = current_input[1]*MAX_TRANS_VEL/speed
            speed = np.sqrt(current_input[0]**2 + current_input[1]**2)
        # record the computed input at time-step t
        input_history[it] = current_input
        speed_history[it] = speed
        min_sensor_history[it] = min_sensor
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
    return state_history, goal_history, input_history, speed_history, min_sensor_history


if __name__ == '__main__':
    
    # Call main computation for robot simulation
    state_history, goal_history, input_history, speed_history, min_sensor_history = simulate_control()


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
    ax.plot(t, min_sensor_history, label='minimum reading distance from the sensor')
    ax.set(xlabel="t [s]", ylabel="distance")
    plt.legend()
    plt.grid()
    plt.title("time series of minimum reading distance from the sensor")

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

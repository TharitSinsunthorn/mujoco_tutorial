import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Using Finite State Machine and Swing up control

xml_path = '/home/tharitto-chan/mujoco_tutorial/mujoco_python/model/simple_pendulum.xml' #xml file (assumes this is in the same folder as this file)
simend = 100 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

FSM_SWINGUP = 0
FSM_HOLD = 1

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    global FSM
    FSM = FSM_SWINGUP

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    global FSM

    if data.qpos[0] >= 2.5 and FSM == FSM_SWINGUP:
        FSM = FSM_HOLD

    if FSM==FSM_SWINGUP:
        set_velocity_servo(2,100)
        data.ctrl[2] = 0.5

    if FSM==FSM_HOLD:
        set_position_servo(1,100)
        set_velocity_servo(2,8)
        data.ctrl[1] = np.pi
    #spring-like position servo
    # set_position_servo(1, 10)
    # data.ctrl[1] = np.pi  #position

    #speed control; velocity servo
    # set_velocity_servo(2,100)
    # data.ctrl[2] = 0.5  #velocity

    #position control; position/velocity servo
    # set_position_servo(1, 100)
    # set_velocity_servo(2,10)
    # data.ctrl[1] = np.pi #position

    #torque control;
    # set_torque_servo(0, 1)
    # data.ctrl[0] = -10*(data.qpos[0]-np.pi)  #torque (spring) [-Kp(@ - @_ref)]
    # data.ctrl[0] = -100*(data.qvel[0]-0.5) #speed control] [-Kv(@_dot - @_dot_ref)]
    # data.ctrl[0] = -100*(data.qpos[0]-np.pi) -10*data.qvel[0] #position control [-Kp(@ - @_ref) - Kv(@_dot)]

def set_torque_servo(actuator_no, flag):
    if (flag==0):
        model.actuator_gainprm[actuator_no, 0] = 0
    else:
        model.actuator_gainprm[actuator_no, 0] = 1

def set_position_servo(actuator_no, kp):
    model.actuator_gainprm[actuator_no, 0] = kp
    model.actuator_biasprm[actuator_no, 1] = -kp

def set_velocity_servo(actuator_no, kv):
    model.actuator_gainprm[actuator_no, 0] = kv
    model.actuator_biasprm[actuator_no, 2] = -kv

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)
    
def plot_debug(EE_pose):
        # Extract coordinates for plotting
        EE_X, EE_Y, EE_Z = zip(*EE_pose)

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot points for each pose
        ax.scatter(EE_X, EE_Y, EE_Z, c='r', marker='o', label='EE Pose')

        #ax.set_xlim([0.2, 0.3])
        # ax.set_ylim([-0.3, 0.3])
        # ax.set_zlim([-self.height, -self.height + 0.5])

        # Set labels and title
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('3D Plot of Poses')
        
        # Add a legend
        ax.legend()

        plt.show()


# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                     # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
cam.azimuth = -90.68741727466428 
cam.elevation = -2.8073894766455036
cam.distance = 5.457557373462702
cam.lookat = np.array([0.0, 0.0, 3.0])

data.qpos[0] = np.pi/2

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)

    if data.time>=simend:
        break

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
#!/usr/bin/env python
import sys
import copy
import rospy
import moveit_commander
import yaml
import time
import os
import csv
import moveit_msgs.msg
import geometry_msgs.msg
from  franka_msgs.msg import FrankaState
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from moveit_msgs.msg import RobotTrajectory
import roslaunch
import scipy
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
import subprocess

#To-do: Run .yaml trajectory with joint_state check, Controller start/end, 
#Low-priority to-do: Franka gripper grasp/move, Biotac calibrate.

class MainProcess(object):
    def __init__(self):
        super(MainProcess, self).__init__()
        rospy.init_node('move_grasp_python', anonymous=True)
        self.filename_pub = rospy.Publisher('/panda_project_control_responce', String, queue_size = 1000) #publishing here saves robotiq+franka+aces+realsense+prophesee data.
        self.target_sub = rospy.Subscriber('/panda_project_control_command', String, self.targetCallback, queue_size=1000)
        self.robotic_states = rospy.Subscriber('/franka_state_controller/franka_states/', FrankaState, self.frankaStatesCallback, queue_size=1000)


        self.lock = False
        self.offset = 0.002 #in m, for 4th movement in data collection, so that item gently or not come into contact with table.
        self.is_moveit_inited = False
        self.start_record = False
        self.positions = []
        self.orientations = []
        self.joint_values = []

    def frankaStatesCallback(self, data):
        if self.start_record:
            ee = data.O_T_EE
            position = [ee[12], ee[13], ee[14]]
            rot_mat  = [ [ee[0], ee[4], ee[8]],
                                       [ee[1], ee[5], ee[9]],
                                       [ee[2], ee[6], ee[10]] ]
            r = self.from_matrix(rot_mat)
            orientation = r.as_quat()
            self.positions.append(position)
            self.orientations.append(orientation)
            self.joint_values.append(data.q)
    
    def start_record_traj(self):
        self.start_record = True
        self.positions = []
        self.orientations = []
        self.joint_values = []

    def stop_record_traj(self):
        self.start_record = False

    def execute_traj(self):
        # run moviet here
        idx = np.floor( np.linspace(0, len(self.positions)-1, 10) ).astype(int)
        print(idx)

        pos = np.array( self.positions )[idx]
        orient = np.array( self.orientations )[idx]
        some_jts = np.array( self.joint_values )[idx]


        # roslaunch panda_moveit_config panda_moveit.launch load_gripper:=true


        # os.spawnl(os.P_NOWAIT, "roslaunch panda_moveit_config panda_moveit.launch load_gripper:=true")
        proc = subprocess.Popen(["roslaunch", "panda_moveit_config", "panda_moveit.launch"])

        print('Running background moviet')
        rospy.sleep(5)

        self.home_routine()
        waypoints = []
        group = self.group
        group.set_max_velocity_scaling_factor(0.1)
        group.set_max_acceleration_scaling_factor(0.05)

        # wpose = group.get_current_pose().pose
        # for i in range(len(pos)):
        #     wpose.position.x = pos[i][0]
        #     wpose.position.y = pos[i][1]
        #     wpose.position.z = pos[i][2]
        #     wpose.orientation.w = orient[i][3]
        #     wpose.orientation.x = orient[i][0]
        #     wpose.orientation.y = orient[i][1]
        #     wpose.orientation.z = orient[i][2]
        #     waypoints.append(copy.deepcopy(wpose))

        # (plan, fraction) = group.compute_cartesian_path(waypoints,
        #                                                 0.02,
        #                                                 0)

        # #print(plan)
        # group.execute(plan)

        for wp in some_jts:
            self.joint_execute(wp[0],wp[1],wp[2],wp[3],wp[4],wp[5],wp[6])

        self.home_routine()

        self.filename_pub.publish("execution_end")
        rospy.sleep(1)

        proc.terminate()
        
        


    def from_matrix(self, matrix):
        is_single = False
        matrix = np.asarray(matrix, dtype=float)

        if matrix.ndim not in [2, 3] or matrix.shape[-2:] != (3, 3):
            raise ValueError("Expected `matrix` to have shape (3, 3) or "
                             "(N, 3, 3), got {}".format(matrix.shape))

        # If a single matrix is given, convert it to 3D 1 x 3 x 3 matrix but
        # set self._single to True so that we can return appropriate objects in
        # the `to_...` methods
        if matrix.shape == (3, 3):
            matrix = matrix.reshape((1, 3, 3))
            is_single = True

        num_rotations = matrix.shape[0]

        decision_matrix = np.empty((num_rotations, 4))
        decision_matrix[:, :3] = matrix.diagonal(axis1=1, axis2=2)
        decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
        choices = decision_matrix.argmax(axis=1)

        quat = np.empty((num_rotations, 4))

        ind = np.nonzero(choices != 3)[0]
        i = choices[ind]
        j = (i + 1) % 3
        k = (j + 1) % 3

        quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
        quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
        quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
        quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

        ind = np.nonzero(choices == 3)[0]
        quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
        quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
        quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
        quat[ind, 3] = 1 + decision_matrix[ind, -1]

        quat /= np.linalg.norm(quat, axis=1)[:, None]

        if is_single:
            return R(quat[0])
        else:
            return R(quat)

    def move_it_init(self):
        if self.is_moveit_inited:
            return
        self.is_moveit_inited = True
        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        ## Instantiate a `RobotCommander`_ object. This object is the outer-level interface to
        ## the robot:
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This object is an interface
        ## to the world surrounding the robot:
        scene = moveit_commander.PlanningSceneInterface()

        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to one group of joints.  In this case the group is the joints in the Panda
        ## arm so we set ``group_name = panda_arm``. If you are using a different robot,
        ## you should change this value to the name of your robot arm planning group.
        ## This interface can be used to plan and execute motions on the Panda:
        group_name = "panda_arm"
        group = moveit_commander.MoveGroupCommander(group_name)

        ## We create a `DisplayTrajectory`_ publisher which is used later to publish
        ## trajectories for RViz to visualize:
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)

        ## END_SUB_TUTORIAL

        ## BEGIN_SUB_TUTORIAL basic_info
        ##
        ## Getting Basic Information
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        planning_frame = group.get_planning_frame()
        print "============ Reference frame: %s" % planning_frame

        # We can also print the name of the end-effector link for this group:
        eef_link = group.get_end_effector_link()
        print "============ End effector: %s" % eef_link

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print "============ Robot Groups:", robot.get_group_names()

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print "============ Printing robot state"
        print robot.get_current_state()
        print ""
        ## END_SUB_TUTORIAL

        self.box_name = ''
        self.robot = robot
        self.scene = scene
        self.group = group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names


    def all_joint_close(self, goal, actual, tolerance):
        """
        Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
        @param: goal       A list of floats, a Pose or a PoseStamped
        @param: actual     A list of floats, a Pose or a PoseStamped
        @param: tolerance  A float
        @returns: bool
        """
        all_equal = True
        if type(goal) is list:
            for index in range(len(goal)):
                if abs(actual[index] - goal[index]) > tolerance:
                    print("ABOVE TOLERANCE")
                    for i in range(len(goal)):
                        print(i, actual[i] , goal[i])
                    return False
                else:
                    print(index, "   ",abs(actual[index] - goal[index]))

        elif type(goal) is geometry_msgs.msg.PoseStamped:
            return self.all_joint_close(goal.pose, actual.pose, tolerance)

        elif type(goal) is geometry_msgs.msg.Pose:
            return self.all_joint_close(pose_to_list(goal), pose_to_list(actual), tolerance)

        return True

    def all_cartesian_close(self, goal, actual, tolerance):
        """
        Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
        @param: goal       A list of floats, a Pose or a PoseStamped
        @param: actual     A list of floats, a Pose or a PoseStamped
        @param: tolerance  A float
        @returns: bool
        """
        all_equal = True
        if type(goal) is list:
            for index in range(7):
                if index <= 2 and abs(actual[index] - goal[index]) > tolerance: #for oosition
                    print("ABOVE TOLERANCE")
                    for i in range(len(goal)):
                        print(i, actual[i] , goal[i])
                    return False
                elif index > 2 and min(abs(actual[index] - goal[index]), abs(actual[index] + goal[index])) > tolerance: #for orientation. q and -q are the same.
                    print("ABOVE TOLERANCE")
                    for i in range(len(goal)):
                        print(i, actual[i] , goal[i])
                    return False
                else:
                    print(index, "   ",abs(actual[index] - goal[index]))

        elif type(goal) is geometry_msgs.msg.PoseStamped:
            return self.all_cartesian_close(goal.pose, actual.pose, tolerance)

        elif type(goal) is geometry_msgs.msg.Pose:
            return self.all_cartesian_close(pose_to_list(goal), pose_to_list(actual), tolerance)

        return True


    def go_to_home(self, robotiq = True):
        if not self.is_moveit_inited:
            print("MoveIt not initialized!")
            return
        group = self.group
        group.set_max_velocity_scaling_factor(0.1)
        joint_goal = group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -pi/4
        joint_goal[2] = 0
        joint_goal[3] = -3*pi/4
        joint_goal[4] = 0
        joint_goal[5] = pi/2
        joint_goal[6] = pi/4 + robotiq * pi/4 # If robotiq 2f-140 is on, go to pi/2, else if it is franka gripper, go to pi/4.
        group.go(joint_goal, wait=True)
        group.stop()
        current_joints = self.group.get_current_joint_values()
        return self.all_joint_close(joint_goal, current_joints, 0.005)


    def joint_execute(self, q1,q2,q3,q4,q5,q6,q7):
        if not self.is_moveit_inited:
            print("MoveIt not initialized!")
            return
        group = self.group
        joint_goal = group.get_current_joint_values()
        joint_goal[0] = q1
        joint_goal[1] = q2
        joint_goal[2] = q3
        joint_goal[3] = q4
        joint_goal[4] = q5
        joint_goal[5] = q6
        joint_goal[6] = q7
        #to_save = group.plan(joint_goal)
        #group.execute(to_save, wait=True)
        #group.stop()
        group.go(joint_goal, wait=True)
        group.clear_pose_targets()





    def targetCallback(self, data):
        print("I heard: ",data.data)
        if self.lock:
            print("LOCKED, ONGOING ROUTINE!")
        elif data.data == "record":
            self.start_record_traj()
        elif data.data == "stop":
            self.stop_record_traj()
        elif data.data == "execute":
            self.execute_traj()
        elif data.data == "home":
            self.home_routine()


    def home_routine(self):
        self.routine_lock()
        if not self.is_moveit_inited:
            self.move_it_init()
        self.go_to_home(robotiq = False)
        self.routine_unlock()


    def routine_lock(self):
        #When a routine is running, it should be locked away from accepting routine requests
        self.lock = True

    def routine_unlock(self):
        #Do this when robot is stationary and completed the routine.
        self.lock = False




def main():
    m_g = MainProcess()
    rospy.spin()

if __name__ == '__main__':
    main()

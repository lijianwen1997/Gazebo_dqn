#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import numpy as np
import math
from math import pi
from std_msgs.msg import Float32MultiArray, Int16
from geometry_msgs.msg import Pose2D
class Env():
    def __init__(self, action_size):
        self.goal_x = 0.5
        self.goal_y = -1
        self.position_x = 0
        self.position_y = 0
        self.theta = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        #self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Float32MultiArray, queue_size=1)
        self.sub_odom = rospy.Subscriber('car_pos', Pose2D, self.getOdometry)
        #self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        #self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        #self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        #self.respawn_goal = Respawn()
        self.activate_pred = False
        self.goal_angle = 0
        self.counter = 0

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position_x, self.goal_y - self.position_y), 2)

        return goal_distance

    def getOdometry(self, odom):
        self.position_x = odom.x
        self.position_y = odom.y
        yaw = odom.theta*0.017
        #orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        #_, _, yaw = euler_from_quaternion(orientation_list)

        self.goal_angle = math.atan2(self.goal_y - self.position_y, self.goal_x - self.position_x)

        heading = self.goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getState(self, scan):
        scan_range = []
        heading = self.heading
        min_range = 0.13
        done = False
        self.activate_pred = False
        # if self.activate_pred == True:
        #     if self.counter < 1:
        #         self.counter = self.counter+1
        #     else:
        #         self.activate_pred = False
        #         self.counter = 0
        for i in range(len(scan.data)):
            if scan.data[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.data[i]):
                scan_range.append(0)

            else:
                if scan.data[i] > 1:
                    scan_range.append(1)
                else:
                    scan_range.append(scan.data[i])
                if scan.data[i] < 0.5:
                    self.activate_pred = True

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)
        if min_range > min(scan_range) > 0:
            done = True

        current_distance = round(math.hypot(self.goal_x - self.position_x, self.goal_y - self.position_y), 2)
        if current_distance < 0.3:
            self.get_goalbox = True
        if self.counter < 10:
            self.counter+=1
        else:
            rospy.loginfo("heading %.2f current_distance %.2f", heading, current_distance)
            self.counter = 0
        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done

    def setReward(self, state, done, action):
        yaw_reward = []
        current_distance = state[-3]
        heading = state[-4]

        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.goal_distance)
        if self.activate_pred:
            reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate)
        else:
            reward = 0
        if done:
            rospy.loginfo("Collision!!")
            reward = -150
            self.pub_cmd_vel.publish(Float32MultiArray())

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 50
            self.pub_cmd_vel.publish(Float32MultiArray())
            self.goal_y = -self.goal_y
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward

    def step(self, action):
        max_angular_vel = 0.25
        #rospy.loginfo(self.activate_pred)
        if self.activate_pred:
            ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5
            rospy.loginfo('____Prediction!! w: %.2f  ', ang_vel)
        else:
            ang_vel = self.heading*0.6/3
            if ang_vel > max_angular_vel:
                ang_vel = max_angular_vel
            if ang_vel < - max_angular_vel:
                ang_vel = -max_angular_vel

            #rospy.loginfo('No Prediction!! w: %.2f  goal_angle: %.2f ',ang_vel,self.heading)

        vel_cmd = Float32MultiArray()
        vel_base = 0.25
        vel_cmd.data = [vel_base - ang_vel , vel_base + ang_vel]
        #vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan_trim', Float32MultiArray, timeout=5)
            except:
                pass

        state, done = self.getState(data)

        reward = self.setReward(state, done, action)

        return np.asarray(state), reward, done

    def reset(self):
        #rospy.wait_for_service('gazebo/reset_simulation')
        #try:
        #    self.reset_proxy()
        #except (rospy.ServiceException) as e:
        #    print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan_trim', Float32MultiArray, timeout=5)
            except:
                pass

        #if self.initGoal:
        #    self.goal_x, self.goal_y = self.respawn_goal.getPosition()
        #    self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)
        vel_cmd = Float32MultiArray()
        vel_cmd.data = [0, 0]
        # vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)
        rospy.sleep(5)
        rospy.loginfo("Sleep!!")
        return np.asarray(state)

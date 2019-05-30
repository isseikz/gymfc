import argparse
import gym
import gymfc
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import math
import os

import math_function as mf
from time import sleep

"""
This script evaluates a PID controller in the GymFC environment. This can be
used as a baseline for comparing to other control algorihtms in GymFC and also
to confirm the GymFC environment is setup and installed correctly.

The PID and mix settings reflect what was used in the following paper,

Koch, William, Renato Mancuso, Richard West, and Azer Bestavros.
"Reinforcement Learning for UAV Attitude Control." arXiv
preprint arXiv:1804.04154 (2018).

For reference, PID

PID Roll = [2, 10, 0.005]
PID PITCH = [10, 10, 0.005]
PID YAW = [4, 50, 0.0]

and mix for values throttle, roll, pitch, yaw,

rear right motor = [ 1.0, -1.0,  0.598, -1.0 ]
front rear motor = [ 1.0, -0.927, -0.598,  1.0 ]
rear left motor  = [ 1.0,  1.0,  0.598,  1.0 ]
front left motor = [ 1.0,  0.927, -0.598, -1.0 ]

PID terms were found first using the Zieglerâ€“Nichols method and then manually tuned
for increased response. The Iris quadcopter does not have an X frame therefore
a custom mixer is required. Using the mesh files found in the Gazebo models they were
imported into a CAD program and the motor constraints were measured. Using these
values the mix calculater found here, https://www.iforce2d.net/mixercalc, was
used to derive the values. The implmementation of the PID controller can be found here,
https://github.com/ivmech/ivPID/blob/master/PID.py, windup has been removed so
another variable was not introduced.
"""


def plot_step_response(desired, actual,
                 end=1., title=None,
                 step_size=0.001, threshold_percent=0.1):
    """
        Args:
            threshold (float): Percent of the start error
    """

    #actual = actual[:,:end,:]
    end_time = len(desired) * step_size
    t = np.arange(0, end_time, step_size)

    #desired = desired[:end]
    threshold = threshold_percent * desired

    plot_min = -math.radians(350)
    plot_max = math.radians(350)

    subplot_index = 3
    num_subplots = 3

    f, ax = plt.subplots(num_subplots, sharex=True, sharey=False)
    f.set_size_inches(10, 5)
    if title:
        plt.suptitle(title)
    ax[0].set_xlim([0, end_time])
    res_linewidth = 2
    linestyles = ["c", "m", "b", "g"]
    reflinestyle = "k--"
    error_linestyle = "r--"

    # Always
    ax[0].set_ylabel("Roll (rad/s)")
    ax[1].set_ylabel("Pitch (rad/s)")
    ax[2].set_ylabel("Yaw (rad/s)")

    ax[-1].set_xlabel("Time (s)")


    """ ROLL """
    # Highlight the starting x axis
    ax[0].axhline(0, color="#AAAAAA")
    ax[0].plot(t, desired[:,0], reflinestyle)
    ax[0].plot(t, desired[:,0] -  threshold[:,0] , error_linestyle, alpha=0.5)
    ax[0].plot(t, desired[:,0] +  threshold[:,0] , error_linestyle, alpha=0.5)

    r = actual[:,0]
    ax[0].plot(t[:len(r)], r, linewidth=res_linewidth)

    ax[0].grid(True)



    """ PITCH """

    ax[1].axhline(0, color="#AAAAAA")
    ax[1].plot(t, desired[:,1], reflinestyle)
    ax[1].plot(t, desired[:,1] -  threshold[:,1] , error_linestyle, alpha=0.5)
    ax[1].plot(t, desired[:,1] +  threshold[:,1] , error_linestyle, alpha=0.5)
    p = actual[:,1]
    ax[1].plot(t[:len(p)],p, linewidth=res_linewidth)
    ax[1].grid(True)


    """ YAW """
    ax[2].axhline(0, color="#AAAAAA")
    ax[2].plot(t, desired[:,2], reflinestyle)
    ax[2].plot(t, desired[:,2] -  threshold[:,2] , error_linestyle, alpha=0.5)
    ax[2].plot(t, desired[:,2] +  threshold[:,2] , error_linestyle, alpha=0.5)
    y = actual[:,2]
    ax[2].plot(t[:len(y)],y , linewidth=res_linewidth)
    ax[2].grid(True)

    plt.show()

class Policy(object):
    def action(self, state, sim_time=0, desired=np.zeros(3), actual=np.zeros(3) ):
        pass
    def reset(self):
        pass

class RandomPolicy(Policy):
    def __init__(self):
        self.cnt = 0
        self.motor_values = np.array([1500]*4)
        self.test_values = np.array([1500]*4)
        self.is_safe = True
        self.update = False
        self.final_reward = 0

    def action(self, state, sim_time=0, actual=np.zeros(6), euler=np.zeros(3), position=np.zeros(3)):
        print(self.is_safe, euler, any([(np.cos(angle)> 1/np.sqrt(2)) for angle in euler]))

        safe_angle = any([np.cos(angle)> 0.98 for angle in euler[0:2]]) # 20 deg
        safe_angvel = np.linalg.norm(actual[3:6]) < 5 # 50 rad/s
        safe_altitude = (position[2] > -0.3)
        safe_landing  = (position[2] > -0.3) & any([np.cos(angle) > 0.985 for angle in euler[0:2]]) # 10deg
        # print(position)

        # TODO: 加速度が明らかにおかしいときはアラート（飛ばない）
        # TODO: 徐々に最大入力を大きくしないと瞬間的なトルクでひっくり返る
        # TODO: 徐々に大きくしながら上に上がり続けられれば報酬を与える？
        # TODO: そして、飛ぶ状態で少しずらした入力でシステム推定
        # motor_values = np.array([1500]*4)
        if (self.is_safe) & (self.cnt % 100 == 0): # update
            print("You Win! Congratulations!")
            self.test_values = np.random.rand(4) * 1000 * sim_time / 20 + 1000
            self.is_safe = False
            self.final_reward = 1
        elif (not self.is_safe) & (safe_angle & safe_landing): # reset
            print("You Lose! Think why you did so!")
            self.is_safe = True
            self.cnt = 0
            self.final_reward = -1
        elif (not safe_angle)|(not safe_angvel)|(not safe_altitude):
            self.is_safe = False
        # print(self.motor_values)

        if self.is_safe:
            self.motor_values = self.test_values
        else:
            self.motor_values = np.zeros(4)
        sleep(0.01)

        self.cnt += 1
            # motor_values = np.array(self.controller.calculate_motor_values(sim_time, desired, actual))
        # Need to scale from 1000-2000 to -1:1
        return np.array( [ (m - 1000)/500  - 1 for m in self.motor_values])

    def reset(self):
        pass

def sensor(env):
    acc = env.linear_acceleration_xyz
    ang = env.omega_actual
    return acc, ang

def get_euler(env):
    quat1234 = env.orientation_quat
    quat0123 = np.array([quat1234[3],quat1234[0],quat1234[1],quat1234[2]])
    return mf.euler_from(quat0123)

def get_position(env):
    return env.position

def eval(env, pi):
    actuals = []
    desireds = []
    pi.reset()
    ob = env.reset()
    while True:
        # desired = env.omega_target
        # actual = env.omega_actual
        # PID only needs to calculate error between desired and actual y_e

        acc, ang = sensor(env)
        euler = get_euler(env)
        pos = get_position(env)
        desired = env.omega_target
        actual = np.hstack((acc, ang))
        ac = pi.action(ob, env.sim_time, actual, euler, pos)
        ob, reward, done, info = env.step(ac)
        actuals.append(actual)
        desireds.append(desired)
        if env.sim_time > 20.0:
            break
    env.close()
    return desireds, actuals


"""
This is essentially a port from Betaflight

For the iris the motors have the following constraints,
https://www.iforce2d.net/mixercalc/
4 2 409
2 1 264
4 3 264
3 1 441
4 1 500
2 3 500

"""

class PIDController(object):
    FD_ROLL = 0
    FD_PITCH = 1
    FD_YAW = 2
    PTERM_SCALE = 0.032029
    ITERM_SCALE = 0.244381
    DTERM_SCALE = 0.000529
    minthrottle = 1070
    maxthrottle = 2000

    def __init__(self, pid_roll = [40, 40, 30], pid_pitch = [58, 50, 35], pid_yaw = [80, 45, 20], itermLimit=150):

        # init gains and scale
        self.Kp = [pid_roll[0], pid_pitch[0], pid_yaw[0]]
        self.Kp = [self.PTERM_SCALE * p for p in self.Kp]

        self.Ki = [pid_roll[1], pid_pitch[1], pid_yaw[1]]
        self.Ki = [self.ITERM_SCALE * i for i in self.Ki]

        self.Kd = [pid_roll[2], pid_pitch[2], pid_yaw[2]]
        self.Kd = [self.DTERM_SCALE * d for d in self.Kd]


        self.itermLimit = itermLimit

        self.previousRateError = [0]*3
        self.previousTime = 0
        self.previous_motor_values = [self.minthrottle]*4
        self.pid_rpy = [PID(*pid_roll), PID(*pid_pitch), PID(*pid_yaw)]

    def calculate_motor_values(self, current_time, sp_rates, gyro_rates):
        rpy_sums = []
        for i in range(3):
            self.pid_rpy[i].SetPoint = sp_rates[i]
            self.pid_rpy[i].update(current_time, gyro_rates[i])
            rpy_sums.append(self.pid_rpy[i].output)
        return self.mix(*rpy_sums)

    def constrainf(self, amt, low, high):
        # From BF src/main/common/maths.h
        if amt < low:
            return low
        elif amt > high:
            return high
        else:
            return amt

    def mix(self, r, p, y):
        PID_MIXER_SCALING = 1000.0
        pidSumLimit = 10000.#500
        pidSumLimitYaw = 100000.#1000.0#400
        motorOutputMixSign = 1
        motorOutputRange = self.maxthrottle - self.minthrottle# throttle max - throttle min
        motorOutputMin = self.minthrottle

        currentMixer=[
            [ 1.0, -1.0,  0.598, -1.0 ],          # REAR_R
            [ 1.0, -0.927, -0.598,  1.0 ],          # RONT_R
            [ 1.0,  1.0,  0.598,  1.0 ],          # REAR_L
            [ 1.0,  0.927, -0.598, -1.0 ],          # RONT_L
        ]
        mixer_index_throttle = 0
        mixer_index_roll = 1
        mixer_index_pitch = 2
        mixer_index_yaw = 3

        scaledAxisPidRoll = self.constrainf(r, -pidSumLimit, pidSumLimit) / PID_MIXER_SCALING
        scaledAxisPidPitch = self.constrainf(p, -pidSumLimit, pidSumLimit) / PID_MIXER_SCALING
        scaledAxisPidYaw = self.constrainf(y, -pidSumLimitYaw, pidSumLimitYaw) / PID_MIXER_SCALING
        scaledAxisPidYaw = -scaledAxisPidYaw

        # Find roll/pitch/yaw desired output
        motor_count = 4
        motorMix = [0]*motor_count
        motorMixMax = 0
        motorMixMin = 0
        # No additional throttle, in air mode
        throttle = 0
        motorRangeMin = 1000
        motorRangeMax = 2000

        for i in range(motor_count):
            mix = (scaledAxisPidRoll  * currentMixer[i][1] +
                scaledAxisPidPitch * currentMixer[i][2] +
                scaledAxisPidYaw   * currentMixer[i][3])

            if mix > motorMixMax:
                motorMixMax = mix
            elif mix < motorMixMin:
                motorMixMin = mix
            motorMix[i] = mix

        motorMixRange = motorMixMax - motorMixMin
        #print("range=", motorMixRange)

        if motorMixRange > 1.0:
            for i in range(motor_count):
                motorMix[i] /= motorMixRange
            # Get the maximum correction by setting offset to center when airmode enabled
            throttle = 0.5

        else:
            # Only automatically adjust throttle when airmode enabled. Airmode logic is always active on high throttle
            throttleLimitOffset = motorMixRange / 2.0
            throttle = self.constrainf(throttle, 0.0 + throttleLimitOffset, 1.0 - throttleLimitOffset)

        motor = []
        for i in range(motor_count):
            motorOutput = motorOutputMin + (motorOutputRange * (motorOutputMixSign * motorMix[i] + throttle * currentMixer[i][mixer_index_throttle]))
            motorOutput = self.constrainf(motorOutput, motorRangeMin, motorRangeMax);
            motor.append(motorOutput)

        motor = list(map(int, np.round(motor)))
        return motor


    def is_airmode_active(self):
        return True

    def reset(self):
        for pid in self.pid_rpy:
            pid.clear()

# This file is part of IvPID.
# Copyright (C) 2015 Ivmech Mechatronics Ltd. <bilgi@ivmech.com>
#
# IvPID is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# IvPID is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# title           :PID.py
# description     :python pid controller
# author          :Caner Durmusoglu
# date            :20151218
# version         :0.1
# notes           :
# python_version  :2.7
# ==============================================================================

"""Ivmech PID Controller is simple implementation of a Proportional-Integral-Derivative (PID) Controller in the Python Programming Language.
More information about PID Controller: http://en.wikipedia.org/wiki/PID_controller
"""
import time

class PID:
    """PID Controller
    """

    def __init__(self, P=0.2, I=0.0, D=0.0):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = 0
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, current_time, feedback_value):
        """Calculates PID value for given reference feedback

        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}

        .. figure:: images/pid_1.png
           :align:   center

           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)

        """
        error = self.SetPoint - feedback_value

        delta_time = current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time =current_time
            self.last_error = error

            #print("P=", self.PTerm, " I=", self.ITerm, " D=", self.DTerm)
            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time

def main(env_id, seed):
    env = gym.make(env_id)
    env.render()
    rank = MPI.COMM_WORLD.Get_rank()
    workerseed = seed + 1000000 * rank
    env.seed(workerseed)
    pi = RandomPolicy()
    desireds, actuals = eval(env, pi)
    title = "PID Step Response in Environment {}".format(env_id)
    plot_step_response(np.array(desireds), np.array(actuals), title=title)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Evaluate a PID controller")
    parser.add_argument('--env-id', help="The Gym environement ID", type=str,
                        default="AttFC_GyroErr-MotorVel_M4_Ep-v0")
    parser.add_argument('--seed', help='RNG seed', type=int, default=17)

    args = parser.parse_args()
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir,
                               "../configs/iris.config")
    print ("Loading config from ", config_path)
    os.environ["GYMFC_CONFIG"] = config_path

    main(args.env_id, args.seed)

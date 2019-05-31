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


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


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

class SafeTakingOffPolicy(object):
    """安全に離陸するための方策を学習するポリシー."""
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

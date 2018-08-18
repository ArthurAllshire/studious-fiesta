import numpy as np
import scipy.optimize
import math
from utils import min_angular_displacement

global dt
dt = 1/10

class Module:
    accel_cap = 1
    delta_v_cap = accel_cap*dt
    delta_theta_cap = 1*dt

    def __init__(self, v_i, theta_i):
        self.v = v_i
        self.theta = theta_i

    def set(self, setpoint):
        constraints_satisfied = [True, True]
        v = np.linalg.norm(setpoint)
        theta = math.atan2(setpoint[1], setpoint[0])
        if abs((v-self.v)) > self.delta_v_cap:
            print('delta_v_cap not satisfied')
            v = np.clip(v, self.v-self.delta_v_cap, self.v+self.delta_v_cap)
            constraints_satisfied[0] = False
        min_angular = min_angular_displacement(self.theta, theta)
        if abs(min_angular) > self.delta_theta_cap:
            print('delta theta not satisfied')
            theta = self.theta + np.clip(min_angular, -self.delta_theta_cap, self.delta_theta_cap)
            constraints_satisfied[1] = False
        self.v = v
        self.theta = theta

def objective(x, v_target):
    return np.linalg.norm(v_target - x)

def jacobian(x, v_target):
    vector = np.zeros(shape=(2))
    obj = objective(x, v_target)
    if obj < 1e-3:
        return vector
    vector[0] = -(v_target[0]-x[0]) / obj
    vector[1] = -(v_target[1]-x[1]) / obj
    return vector

class MovementOptimizer:
    accel_cap = 1
    delta_v_cap = accel_cap*dt
    delta_theta_cap = 1*dt
    max_v = 1

    def __init__(self, module):
        self.module = module
        self.max_v_c = scipy.optimize.NonlinearConstraint(np.linalg.norm, 0, self.max_v)

    def gen_command(self, v_target):
        """Where v_target is [vx, vy] for the robot"""
        def theta_c_fn(x):
            return min_angular_displacement(self.module.theta, math.atan2(x[1], x[0]))
        theta_c = scipy.optimize.NonlinearConstraint(theta_c_fn, -self.delta_theta_cap,
                                                        self.delta_theta_cap)
        def vel_c_fn(x):
            return self.module.v - np.linalg.norm(x)
        vel_c = scipy.optimize.NonlinearConstraint(vel_c_fn, -self.delta_v_cap,
                                                            self.delta_v_cap)

        initial_guess = [self.module.v*math.cos(self.module.theta),
                        self.module.v*math.sin(self.module.theta)]

        res = scipy.optimize.minimize(objective, initial_guess,
                                                   args=(v_target),
                                                   method='trust-constr',
                                                   jac=jacobian,
                                                   hess='2-point',
                                                   constraints=(
                                                   theta_c, vel_c, self.max_v_c))
        command_velocities = res.x
        return command_velocities

class SetpointGenerator:

    def step(self, t, step_vel):
        if t < 1 or t >= 3:
            return np.array([0, 0])
        return step_vel

import numpy as np
import scipy.optimize
import math
from utils import min_angular_displacement

global dt
dt = 1/10

class Module:
    # Set up the physical constraints on the module
    accel_cap = 1
    delta_v_cap = accel_cap*dt
    delta_theta_cap = 1*dt
    max_v = 1

    def __init__(self, v_i, theta_i):
        self.v = v_i
        self.theta = theta_i

    def set(self, setpoint):
        v = np.linalg.norm(setpoint)
        theta = math.atan2(setpoint[1], setpoint[0])
        # constrain the change in the module's delta v, v, and delta theta to the required constraints
        if abs((v-self.v)) > self.delta_v_cap:
            print('Warning: delta_v_cap not satisfied')
            v = np.clip(v, self.v-self.delta_v_cap, self.v+self.delta_v_cap)
        if v > self.max_v:
            print('Warning: set velocity greater than max')
            v = self.max_v
        min_angular = min_angular_displacement(self.theta, theta)
        if abs(min_angular) > self.delta_theta_cap:
            print('Warning: delta_theta_cap not satisfied')
            theta = self.theta + np.clip(min_angular, -self.delta_theta_cap, self.delta_theta_cap)
        self.v = v
        self.theta = theta

def objective(x, v_target):
    # objective funciton for our optimiser to minimise
    return np.linalg.norm(v_target - x)

def jacobian(x, v_target):
    # jacobian of the objective function.
    # of the form [∂/∂x, ∂/∂y]
    vector = np.zeros(shape=(2))
    obj = objective(x, v_target)
    if obj < 1e-3:
        return vector
    vector[0] = -(v_target[0]-x[0]) / obj
    vector[1] = -(v_target[1]-x[1]) / obj
    return vector

class MovementOptimizer:
    # set up the bounds that we use to generate our constraints from
    accel_cap = 1
    delta_v_cap = accel_cap*dt
    delta_theta_cap = 1*dt
    max_v = 1

    def __init__(self, module):
        self.module = module
        # the bound on velocity is just the norm of the [x, y] velocity target vector
        self.max_v_c = scipy.optimize.NonlinearConstraint(np.linalg.norm, 0, self.max_v)

    def gen_command(self, v_target):
        """Where v_target is [vx, vy] for the robot"""

        # create the constraint on our change in theta. this implementation requires that
        # we re-define this function in the loop as it needs access to the module's theta,
        # and there is no way i can find of passing this in without doing this. should find a fix
        def theta_c_fn(x):
            return abs(min_angular_displacement(self.module.theta, math.atan2(x[1], x[0])))
        theta_c = scipy.optimize.NonlinearConstraint(theta_c_fn, 0, self.delta_theta_cap)
        # create the constraint on change in velocity. similarly to the theta change,
        # this needs access to parameters on the module
        def vel_c_fn(x):
            return abs(self.module.v - np.linalg.norm(x))
        vel_c = scipy.optimize.NonlinearConstraint(vel_c_fn, 0, self.delta_v_cap)

        # our initial estimate of x. will be close to our current speed
        initial_guess = [self.module.v*math.cos(self.module.theta),
                        self.module.v*math.sin(self.module.theta)]

        res = scipy.optimize.minimize(objective, initial_guess,
                                                   # arguments to be passed after the state to the
                                                   # objective and jacobian functions
                                                   args=(v_target),
                                                   # use the trust region method to find a solution
                                                   method='trust-constr',
                                                   jac=jacobian,
                                                   # for this implentation we approximate the hessian.
                                                   # when we actually do this on the bot, we will want to compute it
                                                   # analytically but easier for now to approximate
                                                   hess='2-point',
                                                   # pass in the constraints
                                                   constraints=(
                                                   theta_c, vel_c, self.max_v_c))
        command_velocities = res.x
        return command_velocities

class SetpointGenerator:

    def step(self, t, step_vel):
        """Step the velocity setpoint up after a certain amount of time"""
        if t < 1 or t >= 3:
            return np.array([0, 0])
        return step_vel

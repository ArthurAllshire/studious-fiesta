from swerve_optimiser import Module, MovementOptimizer, SetpointGenerator, dt
import numpy as np
import math

sim_time = 5 # s

sp_gen = SetpointGenerator()
module = Module(0, math.pi/4)
optimizer = MovementOptimizer(module)

step_vel = np.array([1, 1])

for t in np.linspace(0, sim_time, num=int(sim_time/dt)+1):
    # generate the setpoint, similarly to how a higher level controller would
    setpoint = sp_gen.step(t, step_vel)
    # find the optimal setpoint that satisfies the constraint
    optimised_sp = optimizer.gen_command(setpoint)
    # send the setpoint we have generated to the module
    module.set(optimised_sp)
    print(f"t {t} Setpoint {setpoint}, Optimized {optimised_sp} module v:{module.v} theta:{module.theta}")

import math

def constrain_angle(angle):
    """Wrap :param angle: to between +pi and -pi"""
    return math.atan2(math.sin(angle), math.cos(angle))

def min_angular_displacement(current, target):
    """Return the minimum (signed) angular displacement to get from :param current:
    to :param target:. In radians."""
    target = constrain_angle(target)
    opp_target = constrain_angle(target + math.pi)
    current = constrain_angle(current)
    diff = constrain_angle(target - current)
    opp_diff = constrain_angle(opp_target - current)

    if abs(diff) < abs(opp_diff):
        return diff
    return opp_diff

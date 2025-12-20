from dataclasses import dataclass

@dataclass
class AxisBacklashState:
    last_dir: int = 0
    skip_left: float = 0.0
    pos_true: float = 0.0
    pos_readback: float = 0.0

def step_with_backlash(target, step, backlash_n, state):
    delta = target - state.pos_true
    if abs(delta) < 1e-12:
        state.pos_readback = target
        return state.pos_true, state.pos_readback
    new_dir = 1 if delta > 0 else -1
    if new_dir != state.last_dir:
        state.last_dir, state.skip_left = new_dir, float(backlash_n)
    move_readback = new_dir * min(abs(delta), step)
    state.pos_readback += move_readback
    if state.skip_left > 0.0:
        if state.skip_left >= 1.0:
            state.skip_left -= 1.0
            move_true = 0.0
        else:
            move_true = (1.0 - state.skip_left) * move_readback
            state.skip_left = 0.0
    else:
        move_true = move_readback
    state.pos_true += move_true
    return state.pos_true, state.pos_readback

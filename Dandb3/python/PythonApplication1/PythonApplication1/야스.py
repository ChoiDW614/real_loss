import numpy as np
    
    delta_q_i = 1
    while 1:
        if np.abs(delta_q_i) <= 0.001:
            break
        C_qi = get_jacobian(C_qt)
        delta_q_i = -C_qi ** (-1) * C_qt
        q_i = q_i + delta_q_i
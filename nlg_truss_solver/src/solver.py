# -*- coding: utf-8 -*-
import numpy as np

class Solver:
    """Numerical solutions for non-linear systems and dynamics."""
    
    def __init__(self, tol=1e-5, max_iter=25):
        self.tol = tol
        self.max_iter = max_iter

    def _solve_linear(self, K, F):
        try:
            return np.linalg.solve(K, F)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(K) @ F

    def riks_arc_length(self, k_func, f_ref, dl0=0.05, max_steps=200, max_disp=7.0):
        u = np.zeros_like(f_ref)
        lam = 0.0
        path_u, path_lam = [u.copy()], [lam]
        K, _ = k_func(u)
        du_Q = self._solve_linear(K, f_ref)
        dlam = dl0
        du = dlam * du_Q
        ds = np.sqrt(np.dot(du, du) + dlam**2)
        limit_lam, limit_u = None, None
        du_prev = du.copy()
        
        for step in range(max_steps):
            u_prev, lam_prev = u.copy(), lam
            if step > 0:
                K, _ = k_func(u)
                du_Q = self._solve_linear(K, f_ref)
                sign = 1.0 if np.dot(du_prev, du_Q) >= 0 else -1.0
                dlam = sign * ds / np.sqrt(np.dot(du_Q, du_Q) + 1.0)
                du = dlam * du_Q
                if sign < 0 and limit_lam is None:
                    limit_lam, limit_u = lam_prev, u_prev.copy()
            
            u += du
            lam += dlam
            du_p, dlam_p = du.copy(), dlam
            
            converged = False
            for _ in range(self.max_iter):
                K, f_int = k_func(u)
                R = lam * f_ref - f_int
                if np.linalg.norm(R) < self.tol:
                    converged = True
                    break
                du_R = self._solve_linear(K, R)
                du_Q = self._solve_linear(K, f_ref)
                d_lam = -np.dot(du_p, du_R) / (np.dot(du_p, du_Q) + dlam_p)
                d_u = du_R + d_lam * du_Q
                u += d_u
                lam += d_lam
                
            if not converged: break
            path_u.append(u.copy())
            path_lam.append(lam)
            du_prev = u - u_prev
            if np.max(np.abs(u)) > max_disp: break
                
        if limit_lam is None:
            limit_lam, limit_u = path_lam[-1], path_u[-1]
        return limit_u, limit_lam, dl0, np.array(path_u), np.array(path_lam)

    def generalized_displacement_control(self, k_func, f_ref, dl0=0.05, max_steps=200, max_disp=7.0):
        u = np.zeros_like(f_ref)
        lam = 0.0
        path_u, path_lam = [u.copy()], [lam]
        u1_first, u1_prev = None, None
        current_dir = 1.0 
        limit_lam, limit_u = None, None

        for step in range(max_steps):
            K, f_int = k_func(u)
            u1_current = self._solve_linear(K, f_ref)

            if step == 0:
                u1_first = u1_prev = u1_current.copy()
                GSP = 1.0
                dlam = dl0 * current_dir
            else:
                den = np.dot(u1_prev, u1_current)
                GSP = np.dot(u1_first, u1_first) / den if den != 0 else 1.0
                if GSP < 0:
                    current_dir *= -1.0  
                    if limit_lam is None:
                        limit_lam, limit_u = lam, u.copy()
                dlam = current_dir * dl0 * np.sqrt(np.abs(GSP))

            u += dlam * u1_current
            lam += dlam

            converged = False
            for _ in range(self.max_iter):
                K, f_int = k_func(u)
                R = lam * f_ref - f_int
                if np.linalg.norm(R) < self.tol:
                    converged = True
                    break
                u1_j = self._solve_linear(K, f_ref)
                u2_j = self._solve_linear(K, R)
                den = np.dot(u1_prev, u1_j)
                if den == 0: break
                dlam_j = -np.dot(u1_prev, u2_j) / den
                u += dlam_j * u1_j + u2_j
                lam += dlam_j

            if not converged: break
            path_u.append(u.copy())
            path_lam.append(lam)
            u1_prev = u1_current.copy()
            if np.max(np.abs(u)) > max_disp: break

        if limit_lam is None:
            limit_lam, limit_u = path_lam[-1], path_u[-1]
        return limit_u, limit_lam, dl0, np.array(path_u), np.array(path_lam)

    def run_dynamics_rk4(self, P_int_func, M, C, P_ext_func, t_max, dt):
        """Solves 1-DOF dynamics using 4th Order Runge-Kutta for snap-through."""
        n_steps = int(t_max / dt)
        times = np.linspace(0, t_max, n_steps)
        w = np.zeros(n_steps)
        v = np.zeros(n_steps)
        
        for i in range(n_steps - 1):
            t = times[i]
            
            def dYdt(t, Y):
                disp, vel = Y[0], Y[1]
                # Equação de movimento: M*a + C*v + Pint(w) = Pext(t)
                accel = (P_ext_func(t) - C * vel - P_int_func(disp)) / M
                return np.array([vel, accel])
            
            Y_curr = np.array([w[i], v[i]])
            k1 = dYdt(t, Y_curr)
            k2 = dYdt(t + dt/2, Y_curr + k1 * dt/2)
            k3 = dYdt(t + dt/2, Y_curr + k2 * dt/2)
            k4 = dYdt(t + dt, Y_curr + k3 * dt)
            
            Y_next = Y_curr + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            w[i+1] = Y_next[0]
            v[i+1] = Y_next[1]
            
        return times, w
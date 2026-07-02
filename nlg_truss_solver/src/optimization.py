# -*- coding: utf-8 -*-
import numpy as np
from src.solver import Solver

# Global configuration constants
MIN_AREA = 0.1

class HrindaOpt:
    """Hrinda Optimizer utilizing SED and arc-length parameters."""
    
    def __init__(self, fea_model, tol=1e-4):
        self.fea = fea_model
        self.tol = tol
        self.solver = Solver()

    def _calculate_ssp(self, iexlk, exlk, dl0, dli, dlk):
        if iexlk == 0: return 1.0
        elif iexlk == 1: return exlk + dl0
        elif iexlk == 2: return exlk + dli
        elif iexlk == 3: return exlk - dlk
        return 1.0

    def optimize(self, max_cycles=50):
        weight_history = []
        num_nodes = len(self.fea.nodes)
        
        f_ext = np.zeros(num_nodes * 3)
        for node_idx, load_vec in self.fea.loads.items():
            f_ext[node_idx*3 : node_idx*3+3] = load_vec
        
        iexlk, exlk = 0, 0.95 
        dl0, dli, dlk = 0.05, 0.05, 0.02
        
        for _ in range(max_cycles):
            u_flat, exlk, _, _, _ = self.solver.generalized_displacement_control(
                self.fea.get_global_system, f_ext, dl0
            )
            u_state = u_flat.reshape(-1, 3)
            
            ssp = self._calculate_ssp(iexlk, exlk, dl0, dli, dlk)
            
            if 0.99 < exlk < 1.0:
                break
                
            sed = self.fea.run(u_state)
            sed_total = np.sum(sed)
            sed_total2 = np.sum(sed**2)
            
            if sed_total2 > 0:
                sed_scale = (sed_total / sed_total2) * (sed**ssp)
                self.fea.areas *= sed_scale
            
            self.fea.areas = np.maximum(self.fea.areas, MIN_AREA)
            
            current_w = np.sum(self.fea.areas * self.fea.rho * self.fea.lengths)
            weight_history.append(current_w)

            if np.max(sed) - np.min(sed) < self.tol:
                self.fea.areas /= exlk
                self.fea.areas = np.maximum(self.fea.areas, MIN_AREA)
                final_w = np.sum(self.fea.areas * self.fea.rho * self.fea.lengths)
                weight_history.append(final_w)
                break
            
        return self.fea.areas, weight_history
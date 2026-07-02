# -*- coding: utf-8 -*-
import numpy as np

class FEA:
    """Finite Element Analysis for 3D geometrically nonlinear trusses."""
    
    def __init__(self, nodes, elements, E, rho, bcs=None, loads=None):
        self.nodes = np.array(nodes, dtype=float)
        self.elements = np.array(elements, dtype=int)
        self.E = E
        self.rho = rho
        self.areas = np.ones(len(elements)) 
        self.bcs = bcs or []      
        self.loads = loads or {}  
        
        self.lengths = np.zeros(len(elements))
        for i, (n1, n2) in enumerate(self.elements):
            self.lengths[i] = np.linalg.norm(self.nodes[n2] - self.nodes[n1])
        
    def _green_strain(self, l0, l_new):
        return (l_new**2 - l0**2) / (2 * l0**2)

    def _tangent_stiffness(self, u, elem_idx):
        n1, n2 = self.elements[elem_idx]
        c1, c2 = self.nodes[n1], self.nodes[n2]
        
        dx, dy, dz = c2 - c1
        l0 = self.lengths[elem_idx]
        
        du = u[n2] - u[n1]
        dx_new = dx + du[0]
        dy_new = dy + du[1]
        dz_new = dz + du[2]
        l_new = np.sqrt(dx_new**2 + dy_new**2 + dz_new**2)
        
        strain = self._green_strain(l0, l_new)
        an = self.E * self.areas[elem_idx] * strain  
        
        C = np.array([-dx_new, -dy_new, -dz_new, dx_new, dy_new, dz_new])
        con1 = (self.E * self.areas[elem_idx]) / l0**3
        k_e = con1 * np.outer(C, C)
        
        k_g_block = np.eye(3) * (an / l0)
        k_g = np.zeros((6, 6))
        k_g[0:3, 0:3] = k_g_block
        k_g[3:6, 3:6] = k_g_block
        k_g[0:3, 3:6] = -k_g_block
        k_g[3:6, 0:3] = -k_g_block
        
        return k_e + k_g, an, l0, l_new, C

    def get_global_system(self, u_flat):
        u = u_flat.reshape(-1, 3)
        num_dofs = len(self.nodes) * 3
        k_global = np.zeros((num_dofs, num_dofs))
        f_int = np.zeros(num_dofs)
        
        for i, (n1, n2) in enumerate(self.elements):
            k_local, an, l0, _, C = self._tangent_stiffness(u, i)
            dofs = [n1*3, n1*3+1, n1*3+2, n2*3, n2*3+1, n2*3+2]
            
            for r in range(6):
                for c in range(6):
                    k_global[dofs[r], dofs[c]] += k_local[r, c]
                    
            f_local = (an / l0) * C
            for r in range(6):
                f_int[dofs[r]] += f_local[r]
                
        penalty = 1e12
        for node_idx, dof_idx in self.bcs:
            idx = node_idx*3 + dof_idx
            k_global[idx, idx] += penalty
            f_int[idx] += penalty * u_flat[idx]
            
        return k_global, f_int

    def _sed(self, l0, l_new, area, strain):
        se = self.E * area * l0 * (strain - ((l_new - l0) / l0))
        return se / (self.rho * l_new * area)

    def run(self, u_state):
        num_elems = len(self.elements)
        sed_list = np.zeros(num_elems)
        for i in range(num_elems):
            _, _, l0, l_new, _ = self._tangent_stiffness(u_state, i)
            strain = self._green_strain(l0, l_new)
            sed_list[i] = self._sed(l0, l_new, self.areas[i], strain)
        return sed_list
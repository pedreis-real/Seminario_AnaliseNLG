# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from src.visualization import Plotter, VtkTrussAnimator
from src.solver import Solver
from src.fea import FEA
from src.optimization import HrindaOpt

class RuntimeExamples:
    """Executes the verification examples."""
    
    def __init__(self):
        self.plotter = Plotter()

    def compare_solvers(self):
        print(" > Executing Solver Comparison (Ex 1)...")
        nodes = [[-100, 0, 0], [0, 2, 0], [100, 0, 0]]
        elements = [[0, 1], [2, 1]]
        bcs = [(0,0), (0,1), (0,2), (2,0), (2,1), (2,2), (1,2), (1,0)]
        loads = {1: [0, -200, 0]}
        
        fea = FEA(nodes, elements, 1.0e7, 0.1, bcs, loads)
        fea.areas = np.array([6.4977, 6.4977])
        f_ext = np.zeros(9)
        f_ext[1*3+1] = -200
        
        solver = Solver(max_iter=30)
        _, _, _, p_u_riks, p_lam_riks = solver.riks_arc_length(
            fea.get_global_system, f_ext, dl0=0.08, max_steps=150, max_disp=6.0
        )
        _, _, _, p_u_gdc, p_lam_gdc = solver.generalized_displacement_control(
            fea.get_global_system, f_ext, dl0=0.08, max_steps=150, max_disp=6.0
        )
        
        self.plotter.plot_solver_comparison(
            p_lam_riks, p_u_riks[:, 1*3+1], p_lam_gdc, p_u_gdc[:, 1*3+1],
            "Comparison: Riks vs GDC (Ex 1)"
        )

    def generate_fig6(self):
        print(" > Generating Fig 6 Path (Ex 1 - GDC)...")
        nodes = [[-100, 0, 0], [0, 2, 0], [100, 0, 0]]
        elements = [[0, 1], [2, 1]]
        bcs = [(0,0), (0,1), (0,2), (2,0), (2,1), (2,2), (1,2), (1,0)]
        loads = {1: [0, -200, 0]}
        
        fea = FEA(nodes, elements, 1.0e7, 0.1, bcs, loads)
        fea.areas = np.array([6.4977, 6.4977])
        f_ext = np.zeros(9)
        f_ext[1*3+1] = -200
        
        solver = Solver(max_iter=30)
        _, _, _, p_u, p_lam = solver.generalized_displacement_control(
            fea.get_global_system, f_ext, dl0=0.08, max_steps=150, max_disp=6.0
        )
        self.plotter.plot_equilibrium_path(p_lam, p_u[:, 1*3+1], "Fig. 6: Two-member truss (GDC)")

    def generate_fig15(self):
        print(" > Generating Fig 15 Path (Ex 3 - GDC)...")
        nodes = [[120, -60, 0], [120, 135, 0], [-60, 135, 0], [-60, -60, 0], [0, 0, 3]]
        elements = [[0, 4], [1, 4], [2, 4], [3, 4]]
        bcs = []
        for i in range(4): bcs.extend([(i, 0), (i, 1), (i, 2)])
        bcs.extend([(4, 0), (4, 1)])
        loads = {4: [0, 0, -200]}
        
        fea = FEA(nodes, elements, 1.0e7, 0.1, bcs, loads)
        fea.areas = np.array([2.1555, 1.6668, 1.7202, 2.0894]) 
        f_ext = np.zeros(15)
        f_ext[4*3+2] = -200
        
        solver = Solver(max_iter=30)
        _, _, _, p_u, p_lam = solver.generalized_displacement_control(
            fea.get_global_system, f_ext, dl0=0.08, max_steps=180, max_disp=7.0
        )
        self.plotter.plot_equilibrium_path(p_lam, p_u[:, 4*3+2], "Fig. 15: Four-member truss (GDC)")

    def example_1_symmetric_truss(self):
        print(" > Running Example 1 Optimization...")
        nodes = [[-100, 0, 0], [0, 2, 0], [100, 0, 0]]
        elements = [[0, 1], [2, 1]]
        bcs = [(0,0), (0,1), (0,2), (2,0), (2,1), (2,2), (1,2), (1,0)]
        fea = FEA(nodes, elements, 1.0e7, 0.1, bcs, {1: [0, -200, 0]})
        fea.areas = np.array([3.00, 5.00])  
        
        opt = HrindaOpt(fea)
        final_areas, w_hist = opt.optimize(max_cycles=15)
        iters = list(range(1, len(w_hist) + 1))
        self.plotter.plot_weight_history(iters, w_hist, "Fig 5: Optimization (Ex 1)")

    def example_2_asymmetric_truss(self):
        print(" > Running Example 2 Optimization...")
        nodes = [[-120, 0, 0], [0, 2, 0], [60, 0, 0]]
        elements = [[0, 1], [2, 1]]
        bcs = [(0,0), (0,1), (0,2), (2,0), (2,1), (2,2), (1,2)]
        fea = FEA(nodes, elements, 1.0e7, 0.1, bcs, {1: [0, -200, 0]})
        fea.areas = np.array([3.00, 5.00]) 
        
        opt = HrindaOpt(fea)
        final_areas, w_hist = opt.optimize(max_cycles=15)
        iters = list(range(1, len(w_hist) + 1))
        self.plotter.plot_weight_history(iters, w_hist, "Fig 10: Optimization (Ex 2)")

    def example_3_four_member_truss(self):
        print(" > Running Example 3 Optimization...")
        nodes = [[120, -60, 0], [120, 135, 0], [-60, 135, 0], [-60, -60, 0], [0, 0, 3]]
        elements = [[0, 4], [1, 4], [2, 4], [3, 4]]
        bcs = []
        for i in range(4): bcs.extend([(i, 0), (i, 1), (i, 2)])
        bcs.extend([(4, 0), (4, 1)])
        fea = FEA(nodes, elements, 1.0e7, 0.1, bcs, {4: [0, 0, -200]})
        fea.areas = np.array([2.20, 2.20, 2.20, 2.20]) 
        
        opt = HrindaOpt(fea)
        final_areas, w_hist = opt.optimize(max_cycles=15)
        iters = list(range(1, len(w_hist) + 1))
        self.plotter.plot_weight_history(iters, w_hist, "Fig 18: Optimization (Ex 3)")

    def example_4_pyramidal_dynamics(self, save_only_gif=False):
        print(" > Running Ex 4: Pyramidal Truss Dynamics (Step vs Ramp)...")
        a, h0, E, rho, A_opt, P_ref = 150.0, 10.0, 10.4e6, 0.101, 0.2393, 200.0
        
        nodes = [[a, a, 0], [-a, a, 0], [-a, -a, 0], [a, -a, 0], [0, 0, h0]]
        elements = [[0, 4], [1, 4], [2, 4], [3, 4]]
        
        # 1. Configuração Dinâmica (Sistema Equivalente 1-DOF)
        L0 = np.sqrt(2*a**2 + h0**2)
        M = (4 * rho * A_opt * L0) / 386.4
        K0 = (2 * E * A_opt / L0**3) * (2 * h0**2)
        C = 2 * 0.02 * np.sqrt(K0 * M)
        
        def P_int(w): return (2 * E * A_opt / L0**3) * (h0 - w) * (2*h0*w - w**2)
        def P_step(t): return P_ref
        def P_ramp(t): return P_ref * (t / 1.0) if t < 1.0 else P_ref
        
        # 2. Resolução RK4
        dt_sim = 0.001
        solver = Solver()
        t_step, w_step = solver.run_dynamics_rk4(P_int, M, C, P_step, t_max=1.5, dt=dt_sim)
        t_ramp, w_ramp = solver.run_dynamics_rk4(P_int, M, C, P_ramp, t_max=1.5, dt=dt_sim)
        
        # 3. Chama a animação 3D VTK (O script pausará aqui até você fechar a janela 3D)
        if save_only_gif:
            # Instancia o animator, mas força o salvamento
            animator = VtkTrussAnimator(nodes, elements, w_step, w_ramp, dt_sim, "Snap-Through")
            animator.export_as_gif("pyramid_snap.gif")
        else:
            self.plotter.animate_3d_truss_vtk(
                nodes, elements, w_step, w_ramp, dt=dt_sim, 
                title="Treliça Piramidal - Snap-Through Dinâmico"
            )
        
        # 4. Gera o Gráfico 2D da Dinâmica
        fig, ax = plt.subplots(figsize=self.plotter.fig_size)
        c1, _ = self.plotter._get_unique_style()
        c2, _ = self.plotter._get_unique_style()
        
        ax.plot(t_step, w_step, color=c1, linestyle='-', linewidth=2, label='Degrau (Rajada)', markevery=2)
        ax.plot(t_ramp, w_ramp, color=c2, linestyle='--', linewidth=2, label='Rampa (Gradual)', markevery=2)
        self.plotter._setup_axes(ax, "Resposta Dinâmica: Snap-Through", "Tempo (s)", "Deslocamento Ápice $w$ (in)")
        ax.legend(fontsize=9, loc='upper left')
        fig.tight_layout()

    def example_5_parametric_h0(self):
        print(" > Running Ex 5: Parametric Stability (h0 variation)...")
        a, E, A_opt = 150.0, 10.4e6, 0.2393
        
        fig, ax = plt.subplots(figsize=self.plotter.fig_size)
        
        for h0 in [5.0, 10.0, 15.0, 20.0]:
            L0 = np.sqrt(2*a**2 + h0**2)
            w = np.linspace(0, 2.5 * h0, 200)
            P_static = (2 * E * A_opt / L0**3) * (h0 - w) * (2*h0*w - w**2)
            
            c, m = self.plotter._get_unique_style()
            ax.plot(w, P_static, color=c, marker=m, markevery=20, markersize=3, linestyle='-', label=f'$h_0$ = {h0} in')
            
        self.plotter._setup_axes(ax, "Interferência de $h_0$ na Estabilidade", "Deslocamento $w$ (in)", "Resistência Estática $P_{int}$ (lb)")
        ax.legend(fontsize=9)
        fig.tight_layout()

    def run_all(self):
        print("\n--- Starting Complete Verification Suite ---\n")
        
        self.compare_solvers()
        self.generate_fig6()
        self.generate_fig15()
        self.example_1_symmetric_truss()
        self.example_2_asymmetric_truss()
        self.example_3_four_member_truss()
        self.example_4_pyramidal_dynamics(
            save_only_gif=False,
        )
        self.example_5_parametric_h0()
        
        print("\n--- Computations Completed. Rendering all separate plots... ---")
        plt.show()
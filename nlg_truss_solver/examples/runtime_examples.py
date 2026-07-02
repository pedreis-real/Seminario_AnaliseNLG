# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from src.visualization import Plotter
from src.solver import Solver
from src.fea import FEA
from src.optimization import HrindaOpt

class RuntimeExamples:
    """Executes the verification examples."""
    
    def __init__(self):
        self.plotter = Plotter()

    def _calculate_prs(self, r_au, area_m2, theta=0.0):
        """
        Calcula a força de pressão de radiação solar (F_PRS) no SI (Newtons).
        :param r_au: Distância heliocêntrica em Unidades Astronômicas (UA).
        :param area_m2: Área da membrana da vela em metros quadrados (m²).
        :param theta: Ângulo de incidência solar em radianos.
        """
        S0 = 1361.0       # Constante Solar em W/m²
        c = 299792458.0   # Velocidade da luz em m/s
        r0 = 1.0          # Distância de referência em UA
        
        S_r = S0 * (r0 / r_au)**2
        F_prs = (2 * S_r * area_m2 * (np.cos(theta)**2)) / c
        return F_prs

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
            "Comparação: Riks vs GDC (Ex 1)"
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
        self.plotter.plot_equilibrium_path(p_lam, p_u[:, 1*3+1], "Fig. 6: 'Two-member truss' (GDC)")

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
        self.plotter.plot_equilibrium_path(p_lam, p_u[:, 4*3+2], "Fig. 15: 'Four-member truss' (GDC)")

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

    def example_4_pyramidal_dynamics(self, r_au=1.0, scale = None, generate_gif = True):
        print(f" > Running Ex 4: Pyramidal Truss Dynamics (Step vs Ramp) | r = {r_au} UA...")
        
        # Parâmetros Ajustados e Consistentes no SI (mm, N, s, ton)
        a, h0, E, A_opt = 2000.0, 50.0, 70000.0, 50.0 
        rho = 2.7e-9  # Densidade do Alumínio em ton/mm³ 
        
        # Área da membrana da vela solar em m² para o cálculo astronômico
        lado_m = (2 * a) / 1000.0
        area_m2 = lado_m**2
        
        F_prs_N = self._calculate_prs(r_au, area_m2)
        
        # --- O SEGREDO DA RESPOSTA BACANA ---
        # Como F_prs_N (~0.0001 N) é irrisória para a treliça, aplicamos um 
        # fator de escala massivo apenas para visualizar a dinâmica do snap-through.
        FATOR_ESCALA = 600000.0 if scale == None else scale
        P_scaled = F_prs_N * FATOR_ESCALA
        
        print("\n" + "-"*50)
        print("   CARGA DA VELA SOLAR (Radiação Escalonada)   ")
        print("-"*50)
        print(f"Distância (r)       : {r_au} UA")
        print(f"Força Real (F_PRS)  : {F_prs_N:.6f} N")
        print(f"Força Escalonada    : {P_scaled:.2f} N (Usada na Simulação)")
        print("-"*50 + "\n")
        
        nodes = [[a, a, 0], [-a, a, 0], [-a, -a, 0], [a, -a, 0], [0, 0, h0]]
        elements = [[0, 4], [1, 4], [2, 4], [3, 4]]
        
        # Configuração Dinâmica
        L0 = np.sqrt(2*a**2 + h0**2)
        M = 4 * rho * A_opt * L0 
        K0 = (2 * E * A_opt / L0**3) * (2 * h0**2)
        C = 2 * 0.02 * np.sqrt(K0 * M)
        
        def P_int(w): return (2 * E * A_opt / L0**3) * (h0 - w) * (2*h0*w - w**2)
        
        # Cargas aplicadas usando a força escalonada
        def P_step(t): return P_scaled if 0.1 < t < 0.8 else 0.0
        def P_ramp(t): return P_scaled * (t / 0.5) if t < 0.5 else P_scaled
        
        # Resolução RK4
        dt_sim = 0.001
        solver = Solver()
        t_step, w_step = solver.run_dynamics_rk4(P_int, M, C, P_step, t_max=1.5, dt=dt_sim)
        t_ramp, w_ramp = solver.run_dynamics_rk4(P_int, M, C, P_ramp, t_max=1.5, dt=dt_sim)
        
        P_step_arr = np.array([P_step(t) for t in t_step])
        P_ramp_arr = np.array([P_ramp(t) for t in t_ramp])
        
        # Geração do GIF 3D Animado (RETORNADO PARA O CELLULOID)
        if generate_gif:
            self.plotter.animate_3d_truss_celluloid(
                nodes, elements, t_step, w_step, P_step_arr, t_ramp, w_ramp, P_ramp_arr, h0, 
                filename="snap_through.gif"
            )
        
        # Gera o Gráfico 2D da Dinâmica
        fig, ax1 = plt.subplots(figsize=self.plotter.fig_size)
        c1, _ = self.plotter._get_unique_style()
        c2, _ = self.plotter._get_unique_style()
        
        l1 = ax1.plot(t_step, w_step, color=c1, linestyle='-', linewidth=2, label='Desloc. $w$ (Degrau)', markevery=2)
        l2 = ax1.plot(t_ramp, w_ramp, color=c2, linestyle='-', linewidth=2, label='Desloc. $w$ (Rampa)', markevery=2)
        self.plotter._setup_axes(ax1, "Resposta Dinâmica e Entradas", "Tempo (s)", "Deslocamento Ápice $w$ (mm)")
        
        ax2 = ax1.twinx()
        l3 = ax2.plot(t_step, P_step_arr, color=c1, linestyle=':', linewidth=2, alpha=0.6, label='Carga $P$ (Degrau)')
        l4 = ax2.plot(t_ramp, P_ramp_arr, color=c2, linestyle=':', linewidth=2, alpha=0.6, label='Carga $P$ (Rampa)')
        ax2.set_ylabel("Carga de Entrada $P(t)$ (N)", fontsize=10)
        
        lns = l1 + l2 + l3 + l4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, fontsize=7, loc='lower left')
        
        fig.tight_layout()

    def example_5_parametric_h0(self):
        print(" > Running Ex 5: Parametric Stability (h0 variation)...")
        # Mantendo os parâmetros sincronizados com o Ex 4
        a, E, A_opt = 2000.0, 70000.0, 50.0 
        
        fig, ax = plt.subplots(figsize=self.plotter.fig_size)
        
        # Ajustado para alturas (h0) proporcionais a um vão de 4 metros (a=2000mm)
        for h0 in [20.0, 40.0, 60.0, 80.0]:
            L0 = np.sqrt(2*a**2 + h0**2)
            w = np.linspace(0, 2.5 * h0, 200)
            P_static = (2 * E * A_opt / L0**3) * (h0 - w) * (2*h0*w - w**2)
            
            c, m = self.plotter._get_unique_style()
            ax.plot(w, P_static, color=c, marker=m, markevery=20, markersize=3, linestyle='-', label=f'$h_0$ = {h0} mm')
            
        self.plotter._setup_axes(ax, "Interferência de $h_0$ na Estabilidade", "Deslocamento $w$ (mm)", "Resistência Estática $P_{int}$ (N)")
        ax.legend(fontsize=9)
        fig.tight_layout()

        plt.show()

    def run_all(self):
        print("\n--- Starting Complete Verification Suite ---\n")
        
        self.compare_solvers()
        self.generate_fig6()
        self.generate_fig15()
        self.example_1_symmetric_truss()
        self.example_2_asymmetric_truss()
        self.example_3_four_member_truss()
        self.example_4_pyramidal_dynamics(r_au=0.307, scale=10000.0, generate_gif=False)
        self.example_5_parametric_h0()
        
        print("\n--- Computations Completed. Rendering all separate plots... ---")
        plt.show()
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v3 as iio
import vtk

# Cores e marcadores globais
GLOBAL_COLORS = list(mcolors.TABLEAU_COLORS.values())
GLOBAL_MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '<', '>']

# Contador global para alternar estilos de forma contínua
GLOBAL_PLOT_COUNTER = 0

class Plotter:
    """Handles all plotting operations with unique predefined styling."""
    
    def __init__(self):
        # Proporção ajustada para janelas individuais
        self.fig_size = (6, 5) 
        self.fig_size_3d = (8, 8)
        
    def _get_unique_style(self):
        """Retorna uma cor e um marcador únicos avançando o contador global."""
        global GLOBAL_PLOT_COUNTER
        c = GLOBAL_COLORS[GLOBAL_PLOT_COUNTER % len(GLOBAL_COLORS)]
        m = GLOBAL_MARKERS[GLOBAL_PLOT_COUNTER % len(GLOBAL_MARKERS)]
        GLOBAL_PLOT_COUNTER += 1
        return c, m

    def _setup_axes(self, ax, title, x_label, y_label):
        """Applies global grid and styling configurations."""
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.grid(which='both', alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        
    def plot_equilibrium_path(self, load_incs, disps, title):
        """Plots the equilibrium path in an isolated window."""
        fig, ax = plt.subplots(figsize=self.fig_size)
        c, m = self._get_unique_style()
        
        # markersize reduzido para 3 e densidade cortada em 50% com markevery=2
        ax.plot(load_incs, disps, color=c, marker=m, linestyle='-', markersize=3, markevery=2)
        
        self._setup_axes(ax, title, 'Load Increment', 'Displacement (in)')
        fig.tight_layout()

    def plot_weight_history(self, iterations, weights, title):
        """Plots the iteration vs weight history in an isolated window."""
        fig, ax = plt.subplots(figsize=self.fig_size)
        c, m = self._get_unique_style()
        
        # markersize reduzido para 3 e densidade cortada em 50%
        ax.plot(iterations, weights, color=c, marker=m, linestyle='--', markersize=3, markevery=2)
        
        self._setup_axes(ax, title, 'Iteration', 'Weight (lb)')
        fig.tight_layout()

    def plot_solver_comparison(self, load1, disp1, load2, disp2, title):
        """Plots two equilibrium paths on the same graph for comparison."""
        fig, ax = plt.subplots(figsize=self.fig_size)
        c1, m1 = self._get_unique_style()
        c2, m2 = self._get_unique_style()
        
        # markersize reduzido para 3 e 2; markevery=2 em ambos
        ax.plot(load1, disp1, color=c1, marker=m1, linestyle='-', markersize=3, markevery=2, label='Riks Arc-Length')
        ax.plot(load2, disp2, color=c2, marker=m2, linestyle='--', markersize=2, markevery=2, label='GDC Method')
        
        self._setup_axes(ax, title, 'Load Increment', 'Displacement (in)')
        ax.legend(fontsize=9)
        fig.tight_layout()

    def animate_3d_truss_imageio(self, nodes, elements, t_step, w_step, P_step, t_ramp, w_ramp, P_ramp, h0, filename="snap_through.gif"):
        """Gera animação super rápida usando imageio e buffers RGB do matplotlib."""
        print(f" > Gerando animação otimizada com imageio: {filename}...")
        
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222, projection='3d')
        ax3 = fig.add_subplot(212)
        
        total_steps = len(w_step)
        
        # Reduz a densidade para cerca de 45 frames totais (muito mais rápido e fluido)
        step_size = max(1, total_steps // 45) 
        
        a = abs(nodes[0][0])
        norm = plt.Normalize(vmin=-h0, vmax=h0)
        
        c_step, _ = self._get_unique_style()
        c_ramp, _ = self._get_unique_style()
        
        frames = []
        
        for i in range(0, total_steps, step_size):
            # 1. LIMPEZA DOS EIXOS (Crucial para não sobrecarregar a RAM)
            ax1.clear()
            ax2.clear()
            ax3.clear()
            
            # 2. RECONSTRUÇÃO DO GRÁFICO 2D (Fundo e Ponto Atual)
            ax3.set_title("Sinais de Entrada: Degrau vs Rampa", fontsize=12, fontweight='bold')
            ax3.set_xlim(0, max(t_step[-1], t_ramp[-1]))
            ax3.set_ylim(0, max(np.max(P_step), np.max(P_ramp)) * 1.2)
            ax3.set_xlabel("Tempo (s)", fontsize=10)
            ax3.set_ylabel("Carga de Entrada $P(t)$ (lb)", fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            # Linhas base
            ax3.plot(t_step, P_step, color=c_step, linestyle='-', alpha=0.3, label='Degrau (Rajada)')
            ax3.plot(t_ramp, P_ramp, color=c_ramp, linestyle='--', alpha=0.3, label='Rampa (Gradual)')
            ax3.legend(loc='lower right')
            
            # Ponto do instante de tempo atual
            ax3.plot(t_step[i], P_step[i], marker='o', color=c_step, markersize=8)
            ax3.plot(t_ramp[i], P_ramp[i], marker='o', color=c_ramp, markersize=8)
            
            # 3. RECONSTRUÇÃO DOS GRÁFICOS 3D
            def draw_frame(ax, w_val, title):
                base_z = -h0 + w_val
                color = cm.coolwarm(norm(base_z))
                
                # Vela (Z=0 fixo)
                verts_vela = [[(a, a, 0), (-a, a, 0), (-a, -a, 0), (a, -a, 0)]]
                vela = Poly3DCollection(verts_vela, alpha=0.2, facecolor='gray', edgecolor='gray', linestyles='--')
                ax.add_collection3d(vela)
                
                # Base Móvel
                verts_base = [[(a, a, base_z), (-a, a, base_z), (-a, -a, base_z), (a, -a, base_z)]]
                base_poly = Poly3DCollection(verts_base, alpha=0.3, facecolor=color, edgecolor='none')
                ax.add_collection3d(base_poly)
                
                # Barras
                apex = [0, 0, 0]
                nodes_base = [[a, a, base_z], [-a, a, base_z], [-a, -a, base_z], [a, -a, base_z]]
                
                for j in range(4):
                    p1 = nodes_base[j]
                    p2 = apex
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black', linewidth=2.0)
                    
                # Nós móveis (Apoios) e Nó fixo (Central)
                xs, ys, zs = [n[0] for n in nodes_base], [n[1] for n in nodes_base], [n[2] for n in nodes_base]
                ax.scatter(xs, ys, zs, color=color, edgecolor='black', s=60, depthshade=False, zorder=5)
                ax.scatter([0], [0], [0], color='white', edgecolor='black', s=100, depthshade=False, zorder=6)
                
                # Configurações de Câmera e Limites
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_xlim(-a*1.2, a*1.2)
                ax.set_ylim(-a*1.2, a*1.2)
                ax.set_zlim(-h0*1.5, h0*1.5)
                ax.view_init(elev=15, azim=-30)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

            draw_frame(ax1, w_step[i], "Snap-Through: Degrau")
            draw_frame(ax2, w_ramp[i], "Snap-Through: Rampa")
            
            fig.tight_layout()
            
            # 4. EXTRAÇÃO DIRETA DA IMAGEM PARA A RAM (Sem salvar em disco)
            fig.canvas.draw()
            
            # buffer_rgba retorna um array (height, width, 4). Pegamos os 3 primeiros canais (RGB)
            img_rgba = np.asarray(fig.canvas.buffer_rgba())
            img_rgb = img_rgba[:, :, :3] 
            
            frames.append(img_rgb)
        
        plt.close(fig)
        
        # 5. SALVAMENTO NATIVO VIA IMAGEIO (Muito mais rápido)
        fps = 15
        duration = 1000 / fps  # imageio.v3 espera duração em milissegundos
        frames_array = np.array(frames)
        iio.imwrite(filename, frames_array, duration=duration, loop=0)
        
        print(f" > GIF salvo com sucesso em {filename}")
    
    def animate_3d_truss_celluloid(self, nodes, elements, t_step, w_step, P_step, t_ramp, w_ramp, P_ramp, h0, filename="snap_through.gif"):
        """Gera uma animação com o nó central fixo, apoios móveis e gráfico de entrada simultâneo."""
        print(f" > Gerando animação com Celluloid: {filename}... (Isso pode levar alguns segundos)")
        
        from celluloid import Camera
        # Aumentamos o tamanho da figura para caber o gráfico de sinal
        fig = plt.figure(figsize=(10, 10))
        
        # Top row: Treliças 3D
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222, projection='3d')
        # Bottom row: Gráfico das entradas de carga P(t) ocupando toda a largura
        ax3 = fig.add_subplot(212)
        
        camera = Camera(fig)
        
        total_steps = len(w_step)
        step_size = max(1, total_steps // 100) # Mantém ~100 frames
        
        a = abs(nodes[0][0])
        norm = plt.Normalize(vmin=-h0, vmax=h0)
        
        # --- PREPARAÇÃO DO GRÁFICO 2D (Pano de Fundo Estático) ---
        ax3.set_title("Sinais de Entrada: Degrau vs Rampa", fontsize=12, fontweight='bold')
        ax3.set_xlim(0, max(t_step[-1], t_ramp[-1]))
        ax3.set_ylim(0, max(np.max(P_step), np.max(P_ramp)) * 1.2)
        ax3.set_xlabel("Tempo (s)", fontsize=10)
        ax3.set_ylabel("Carga de Entrada $P(t)$ (lb)", fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        c_step, _ = self._get_unique_style()
        c_ramp, _ = self._get_unique_style()
        
        # Plotamos as linhas inteiras antes do loop. O Celluloid as manterá como "background".
        ax3.plot(t_step, P_step, color=c_step, linestyle='-', alpha=0.3, label='Degrau (Rajada)')
        ax3.plot(t_ramp, P_ramp, color=c_ramp, linestyle='--', alpha=0.3, label='Rampa (Gradual)')
        ax3.legend(loc='lower right')
        
        # --- FUNÇÃO DE DESENHO 3D ---
        def draw_frame(ax, w_val, title):
            # Cinemática invertida: Ápice fixo em z=0. A base move-se de -h0 para cima.
            base_z = -h0 + w_val
            color = cm.coolwarm(norm(base_z))
            
            # 1. Vela Solar (PARADA no z=0, coincidindo com o ápice)
            verts_vela = [[(a, a, 0), (-a, a, 0), (-a, -a, 0), (a, -a, 0)]]
            vela = Poly3DCollection(verts_vela, alpha=0.2, facecolor='gray', edgecolor='gray', linestyles='--')
            ax.add_collection3d(vela)
            
            # 2. Superfície Virtual da Base Móvel
            verts_base = [[(a, a, base_z), (-a, a, base_z), (-a, -a, base_z), (a, -a, base_z)]]
            base_poly = Poly3DCollection(verts_base, alpha=0.3, facecolor=color, edgecolor='none')
            ax.add_collection3d(base_poly)
            
            # 3. Barras (conectando base móvel ao ápice fixo)
            apex = [0, 0, 0]
            nodes_base = [[a, a, base_z], [-a, a, base_z], [-a, -a, base_z], [a, -a, base_z]]
            
            for i in range(4):
                p1 = nodes_base[i]
                p2 = apex
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black', linewidth=2.0)
                
            # 4. Nós dos Apoios (Móveis)
            xs = [n[0] for n in nodes_base]
            ys = [n[1] for n in nodes_base]
            zs = [n[2] for n in nodes_base]
            ax.scatter(xs, ys, zs, color=color, edgecolor='black', s=60, depthshade=False, zorder=5)
            
            # 5. Nó Central (Fixo no z=0)
            ax.scatter([0], [0], [0], color='white', edgecolor='black', s=100, depthshade=False, zorder=6)
            
            # Configurações do plot 3D
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlim(-a*1.2, a*1.2)
            ax.set_ylim(-a*1.2, a*1.2)
            ax.set_zlim(-h0*1.5, h0*1.5)
            ax.view_init(elev=15, azim=-30) # Azimute alterado para -30
            ax.set_xlabel('Eixo X')
            ax.set_ylabel('Eixo Y')
            ax.set_zlabel('Eixo Z')

        # --- LOOP DE ANIMAÇÃO ---
        for i in range(0, total_steps, step_size):
            # Desenha os frames 3D
            draw_frame(ax1, w_step[i], "Snap-Through: Degrau")
            draw_frame(ax2, w_ramp[i], "Snap-Through: Rampa")
            
            # Desenha os pontos móveis no gráfico 2D
            ax3.plot(t_step[i], P_step[i], marker='o', color=c_step, markersize=8)
            ax3.plot(t_ramp[i], P_ramp[i], marker='o', color=c_ramp, markersize=8)
            
            fig.tight_layout()
            camera.snap()
            
        anim = camera.animate(blit=False)
        anim.save(filename, writer='pillow', fps=15)
        print(f" > GIF salvo com sucesso em {filename}")
        plt.close(fig)
    
    def animate_3d_truss_vtk(self, nodes, elements, w_step, w_ramp, dt, title):
        """Gera a animação 3D interativa usando VTK."""
        animator = VtkTrussAnimator(nodes, elements, w_step, w_ramp, dt, title)
        animator.start()


class VtkTrussAnimator:
    """Gerencia a renderização 3D e animação temporal da treliça com VTK."""
    
    def __init__(self, nodes, elements, w_step, w_ramp, dt, title):
        self.nodes = np.array(nodes)
        self.elements = elements
        self.w_step = w_step
        self.w_ramp = w_ramp
        self.dt = dt
        self.title = title
        
        # Controle de Animação
        self.current_w = self.w_step
        self.mode_name = "Degrau (Rajada)"
        self.timer_count = 0
        self.is_playing = False
        self.h0 = self.nodes[4][2] # Z inicial do ápice
        
        # Passo de tempo ajustado para rodar perto da velocidade real (60 FPS base)
        self.steps_per_frame = max(1, int(16 / (self.dt * 1000))) 
        
        self.setup_scene()
        
    def setup_scene(self):
        self.points = vtk.vtkPoints()
        for n in self.nodes:
            self.points.InsertNextPoint(n[0], n[1], n[2])
            
        # O quadrado da vela solar agora nasce centralizado na altura (h0/2)
        self.sail_points = vtk.vtkPoints()
        for i in range(4):
            n = self.nodes[i]
            self.sail_points.InsertNextPoint(n[0], n[1], self.h0 / 2.0)
            
        lines = vtk.vtkCellArray()
        for e in self.elements:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, e[0])
            line.GetPointIds().SetId(1, e[1])
            lines.InsertNextCell(line)
            
        self.truss_poly = vtk.vtkPolyData()
        self.truss_poly.SetPoints(self.points)
        self.truss_poly.SetLines(lines)
        
        truss_mapper = vtk.vtkPolyDataMapper()
        truss_mapper.SetInputData(self.truss_poly)
        self.truss_actor = vtk.vtkActor()
        self.truss_actor.SetMapper(truss_mapper)
        self.truss_actor.GetProperty().SetColor(0.2, 0.2, 0.2)
        self.truss_actor.GetProperty().SetLineWidth(4.0)
        
        # Esferas para os Nós
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(3.5)
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(self.truss_poly)
        glyph.SetSourceConnection(sphere.GetOutputPort())
        node_mapper = vtk.vtkPolyDataMapper()
        node_mapper.SetInputConnection(glyph.GetOutputPort())
        self.node_actor = vtk.vtkActor()
        self.node_actor.SetMapper(node_mapper)
        self.node_actor.GetProperty().SetColor(0.8, 0.1, 0.1)
        
        # Vela Solar (Superfície Translúcida)
        sail_polygon = vtk.vtkPolygon()
        sail_polygon.GetPointIds().SetNumberOfIds(4)
        for i in range(4):
            sail_polygon.GetPointIds().SetId(i, i)
        sail_cells = vtk.vtkCellArray()
        sail_cells.InsertNextCell(sail_polygon)
        self.sail_polydata = vtk.vtkPolyData()
        self.sail_polydata.SetPoints(self.sail_points)
        self.sail_polydata.SetPolys(sail_cells)
        
        sail_mapper = vtk.vtkPolyDataMapper()
        sail_mapper.SetInputData(self.sail_polydata)
        self.sail_actor = vtk.vtkActor()
        self.sail_actor.SetMapper(sail_mapper)
        self.sail_actor.GetProperty().SetColor(0.12, 0.47, 0.71)
        self.sail_actor.GetProperty().SetOpacity(0.4) # Transparência
        
        # Painel de Texto de UI
        self.text_actor = vtk.vtkTextActor()
        self.update_text()
        self.text_actor.GetTextProperty().SetFontSize(18)
        self.text_actor.GetTextProperty().SetColor(0, 0, 0)
        self.text_actor.SetPosition(20, 20)
        
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.95, 0.95, 0.95)
        self.renderer.AddActor(self.truss_actor)
        self.renderer.AddActor(self.node_actor)
        self.renderer.AddActor(self.sail_actor)
        self.renderer.AddActor(self.text_actor)
        
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetSize(800, 800)
        self.render_window.SetWindowName(self.title)
        self.render_window.AddRenderer(self.renderer)
        
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.render_window)
        
        # Setup de Câmera isométrica aproximada
        self.renderer.ResetCamera()
        cam = self.renderer.GetActiveCamera()
        cam.Elevation(15)
        cam.Azimuth(45)
        
        self.iren.AddObserver('KeyPressEvent', self.on_key_press)
        self.iren.AddObserver('TimerEvent', self.on_timer)
        self.timer_id = self.iren.CreateRepeatingTimer(16) # ~60 FPS
        
    def update_text(self):
        status = "EXECUTANDO" if self.is_playing else "PAUSADO"
        self.text_actor.SetInput(
            f"Vela Solar: Centralizada dinamicamente\n"
            f"Modo Atual: {self.mode_name} | {status}\n\n"
            f"[ESPACO] Play / Pause\n"
            f"[ M ] Alternar Dinâmica (Degrau/Rampa)\n"
            f"[ R ] Reiniciar Animação"
        )
        
    def on_key_press(self, obj, event):
        key = obj.GetKeySym()
        if key == 'space':
            self.is_playing = not self.is_playing
        elif key == 'm' or key == 'M':
            if self.mode_name == "Degrau (Rajada)":
                self.mode_name = "Rampa (Gradual)"
                self.current_w = self.w_ramp
            else:
                self.mode_name = "Degrau (Rajada)"
                self.current_w = self.w_step
            self.timer_count = 0
        elif key == 'r' or key == 'R':
            self.timer_count = 0
            
        self.update_text()
        self.render_window.Render()
        
    def on_timer(self, obj, event):
        if not self.is_playing:
            return
            
        # Garante que a animação ande em "real-time" se o dt numérico for muito pequeno
        self.timer_count += self.steps_per_frame 
        idx = self.timer_count % len(self.current_w)
        w = self.current_w[idx]
        
        # Deslocamento do ápice (Snap-Through Z)
        new_apex_z = self.h0 - w
        self.points.SetPoint(4, 0.0, 0.0, new_apex_z)
        self.points.Modified()
        
        # A vela solar acompanha o ápice pela metade para ficar sempre centralizada
        new_sail_z = new_apex_z / 2.0
        for i in range(4):
            n = self.nodes[i]
            self.sail_points.SetPoint(i, n[0], n[1], new_sail_z)
        self.sail_points.Modified()
        
        self.render_window.Render()
        
    def start(self):
        print(" [!] Janela VTK Aberta. Interaja usando ESPAÇO e M.")
        print(" [!] Feche a janela 3D para continuar a execução do script.")
        self.iren.Initialize()
        self.render_window.Render()
        self.iren.Start()
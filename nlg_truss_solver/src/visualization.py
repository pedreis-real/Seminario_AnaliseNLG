# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vtk
import imageio.v3 as iio

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
            
        # Pontos da vela solar
        self.sail_points = vtk.vtkPoints()
        for i in range(4):
            n = self.nodes[i]
            self.sail_points.InsertNextPoint(n[0], n[1], 0.0)
            
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
        
        self.render_window.Render()
        
    def start(self):
        print(" [!] Janela VTK Aberta. Interaja usando ESPAÇO e M.")
        print(" [!] Feche a janela 3D para continuar a execução do script.")
        self.iren.Initialize()
        self.render_window.Render()
        self.iren.Start()
    
    def export_as_gif(self, filename="snap_through.gif", total_frames=100):
        """Renderiza a animação para um arquivo .gif sem abrir janela."""
        print(f" > Gerando GIF via imageio: {filename}...")
        
        frames = []
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(self.render_window)
        window_to_image.SetInputBufferTypeToRGB()
        
        self.render_window.OffScreenRenderingOn()
        self.render_window.Render()
        
        for i in range(total_frames):
            # Lógica de atualização dos pontos
            w = self.current_w[int(i * (len(self.current_w)/total_frames))]
            self.points.SetPoint(4, 0.0, 0.0, self.h0 - w)
            self.points.Modified()
            
            self.render_window.Render()
            
            # Captura o buffer da janela VTK
            window_to_image.Modified()
            data = window_to_image.GetOutput()
            width, height = data.GetDimensions()
            
            # Converte para array numpy
            vtk_array = data.GetPointData().GetScalars()
            img_array = np.frombuffer(vtk_array, dtype=np.uint8).reshape(height, width, 3)
            frames.append(np.flipud(img_array)) # VTK inverte o eixo Y em relação ao numpy
            
        # Salva o GIF
        iio.imwrite(filename, frames, duration=100, loop=0)
        print(f" > GIF salvo com sucesso!")
import sys
import time
import numpy as np
import matplotlib
import csv
from datetime import datetime
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox,
    QCheckBox, QScrollArea, QTableWidget, QTableWidgetItem,
    QFileDialog, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon

class TSPSolverApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced TSP Solver")
        self.setWindowIcon(QIcon('icon.png'))  # Add your icon file
        self.setGeometry(100, 100, 1400, 900)
        self.num_cities = 20
        self.dark_mode = False
        self.results = {}
        
        # Custom color palette
        self.primary_color = QColor(41, 128, 185)
        self.secondary_color = QColor(52, 152, 219)
        self.dark_bg = QColor(44, 47, 51)
        self.dark_text = QColor(220, 220, 220)
        
        self.init_ui()
        self.apply_styles()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Header
        header = QLabel("Traveling Salesman Problem Solver")
        header.setFont(QFont('Arial', 16, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)
        
        # Control Panel
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout()
        
        self.city_entry = QLineEdit(str(self.num_cities))
        self.city_entry.setFixedWidth(80)
        self.city_entry.setFont(QFont('Arial', 10))
        
        self.set_cities_btn = self.create_button("Set Cities", self.update_cities)
        self.run_ga_btn = self.create_button("Run GA", self.run_ga)
        self.run_sa_btn = self.create_button("Run SA", self.run_sa)
        self.run_pso_btn = self.create_button("Run PSO", self.run_pso)
        self.compare_btn = self.create_button("Compare", self.compare_algorithms)
        self.save_btn = self.create_button("Save Data", self.save_comparison)
        self.dark_mode_check = QCheckBox("Dark Mode")
        self.dark_mode_check.stateChanged.connect(self.toggle_dark_mode)
        
        control_layout.addWidget(QLabel("Cities:"))
        control_layout.addWidget(self.city_entry)
        control_layout.addWidget(self.set_cities_btn)
        control_layout.addWidget(self.run_ga_btn)
        control_layout.addWidget(self.run_sa_btn)
        control_layout.addWidget(self.run_pso_btn)
        control_layout.addWidget(self.compare_btn)
        control_layout.addWidget(self.save_btn)
        control_layout.addWidget(self.dark_mode_check)
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Visualization Area
        vis_group = QGroupBox("Solutions Visualization")
        vis_layout = QVBoxLayout()
        
        self.fig, self.axs = plt.subplots(1, 3, figsize=(12, 5))
        self.fig.suptitle("Algorithm Solutions Comparison", fontsize=12)
        self.canvas = FigureCanvas(self.fig)
        vis_layout.addWidget(self.canvas)
        vis_group.setLayout(vis_layout)
        main_layout.addWidget(vis_group)
        
        # Results Area
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(["Algorithm", "Distance", "Time (s)", "Iterations", "Cities"])
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        results_layout.addWidget(self.results_table)
        
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        # Status Bar
        self.statusBar().showMessage("Ready")
        
        self.set_cities(self.num_cities)
        
    def create_button(self, text, callback):
        btn = QPushButton(text)
        btn.setFont(QFont('Arial', 10))
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.primary_color.name()};
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {self.secondary_color.name()};
            }}
        """)
        btn.clicked.connect(callback)
        return btn
        
    def apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {'#2c3e50' if self.dark_mode else '#f5f5f5'};
                color: {'white' if self.dark_mode else 'black'};
            }}
            QGroupBox {{
                border: 1px solid gray;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }}
            QTableWidget {{
                border: 1px solid gray;
                gridline-color: gray;
            }}
            QHeaderView::section {{
                background-color: {self.primary_color.name()};
                color: white;
                padding: 4px;
            }}
        """)
        
    def toggle_dark_mode(self):
        self.dark_mode = self.dark_mode_check.isChecked()
        self.apply_styles()
        self.plot_cities()
        
    def set_cities(self, n):
        self.cities = self.generate_cities(n)
        self.dist_matrix = self.compute_distance_matrix(self.cities)
        self.clear_graphs()
        self.results = {}
        self.plot_cities()
        self.statusBar().showMessage(f"Generated {n} random cities")
        
    def generate_cities(self, n=20, bounds=(0, 100)):
        np.random.seed()
        return np.random.uniform(bounds[0], bounds[1], (n, 2))
        
    def compute_distance_matrix(self, cities):
        n = len(cities)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i, j] = np.linalg.norm(cities[i] - cities[j])
        return matrix
        
    def clear_graphs(self):
        for ax in self.axs:
            ax.clear()
            ax.grid(True)
        self.canvas.draw()
        
    def plot_cities(self):
        for ax in self.axs:
            ax.clear()
            ax.scatter(self.cities[:, 0], self.cities[:, 1], c='blue', marker='o')
            for i, (x, y) in enumerate(self.cities):
                ax.text(x, y, str(i+1), fontsize=10, 
                       color='white' if self.dark_mode else 'black',
                       ha='right')
            ax.grid(True)
        self.canvas.draw()
        
    def update_cities(self):
        try:
            n = int(self.city_entry.text())
            if n <= 0:
                raise ValueError
            self.set_cities(n)
        except ValueError:
            QMessageBox.critical(self, "Error", "Please enter a valid positive integer.")
            
    def run_ga(self):
        self.statusBar().showMessage("Running Genetic Algorithm...")
        QApplication.processEvents()
        
        best_cost, best_path, exec_time, iterations = self.run_genetic_algorithm(self.dist_matrix)
        self.results["GA"] = (best_cost, best_path, exec_time, iterations)
        self.plot_graph(self.axs[0], f"GA: {best_cost:.2f}", best_path)
        self.update_results_table()
        self.statusBar().showMessage(f"GA completed in {exec_time:.2f}s - Distance: {best_cost:.2f}")
        
    def run_sa(self):
        self.statusBar().showMessage("Running Simulated Annealing...")
        QApplication.processEvents()
        
        best_cost, best_path, exec_time, iterations = self.run_simulated_annealing(self.dist_matrix)
        self.results["SA"] = (best_cost, best_path, exec_time, iterations)
        self.plot_graph(self.axs[1], f"SA: {best_cost:.2f}", best_path)
        self.update_results_table()
        self.statusBar().showMessage(f"SA completed in {exec_time:.2f}s - Distance: {best_cost:.2f}")
        
    def run_pso(self):
        self.statusBar().showMessage("Running Particle Swarm Optimization...")
        QApplication.processEvents()
        
        best_cost, best_path, exec_time, iterations = self.run_particle_swarm_optimization(self.dist_matrix)
        self.results["PSO"] = (best_cost, best_path, exec_time, iterations)
        self.plot_graph(self.axs[2], f"PSO: {best_cost:.2f}", best_path)
        self.update_results_table()
        self.statusBar().showMessage(f"PSO completed in {exec_time:.2f}s - Distance: {best_cost:.2f}")
        
    def plot_graph(self, ax, title, path):
        ax.clear()
        cities = self.cities
        ax.scatter(cities[:, 0], cities[:, 1], c='blue', marker='o')
        for i, (x, y) in enumerate(cities):
            ax.text(x, y, str(i+1), fontsize=10, 
                   color='white' if self.dark_mode else 'black',
                   ha='right')
        
        for i in range(len(path)):
            x1, y1 = cities[path[i]]
            x2, y2 = cities[path[(i+1) % len(path)]]
            ax.plot([x1, x2], [y1, y2], 'r-', linewidth=1)
            
        ax.set_title(title, color='white' if self.dark_mode else 'black')
        ax.grid(True)
        self.canvas.draw()
        
    def update_results_table(self):
        self.results_table.setRowCount(len(self.results))
        
        for row, (algo, (cost, path, time, iters)) in enumerate(self.results.items()):
            self.results_table.setItem(row, 0, QTableWidgetItem(algo))
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{cost:.2f}"))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{time:.2f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(str(iters)))
            self.results_table.setItem(row, 4, QTableWidgetItem(str(len(self.cities))))
            
        self.results_table.resizeColumnsToContents()
        
    def compare_algorithms(self):
        if not self.results:
            QMessageBox.warning(self, "Warning", "Run at least one algorithm first!")
            return
            
        comparison = {}
        for algo, (cost, _, time, iters) in self.results.items():
            comparison[algo] = {
                'Distance': cost,
                'Time': time,
                'Iterations': iters
            }
            
        msg = "Algorithm Comparison:\n\n"
        for metric in ['Distance', 'Time', 'Iterations']:
            sorted_algs = sorted(comparison.items(), key=lambda x: x[1][metric])
            msg += f"{metric}:\n"
            for i, (alg, data) in enumerate(sorted_algs):
                msg += f"  {i+1}. {alg}: {data[metric]:.2f}\n"
            msg += "\n"
            
        QMessageBox.information(self, "Comparison Results", msg)
        
    def save_comparison(self):
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results to save!")
            return
            
        options = QFileDialog.Option()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Comparison Data", "", 
            "CSV Files (*.csv);;All Files (*)", 
            options=options)
            
        if file_name:
            try:
                with open(file_name, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Algorithm', 'Distance', 'Time (s)', 'Iterations', 'Cities', 'Timestamp'])
                    
                    for algo, (cost, _, time, iters) in self.results.items():
                        writer.writerow([
                            algo,
                            f"{cost:.2f}",
                            f"{time:.2f}",
                            iters,
                            len(self.cities),
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ])
                        
                QMessageBox.information(self, "Success", f"Results saved to {file_name}")
                self.statusBar().showMessage(f"Results saved to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
                
    def run_genetic_algorithm(self, dist_matrix, population_size=100, max_generations=1000, mutation_rate=0.1, early_stopping=50):
        start_time = time.time()
        n = len(dist_matrix)
        population = [np.random.permutation(n) for _ in range(population_size)]
        best_cost = float('inf')
        best_path = None
        no_improvement = 0
        
        def fitness(path):
            return sum(dist_matrix[path[i], path[(i+1) % n]] for i in range(n))
            
        for gen in range(max_generations):
            pop_fitness = [(ind, fitness(ind)) for ind in population]
            pop_fitness.sort(key=lambda x: x[1])
            population = [ind for ind, _ in pop_fitness]
            
            if pop_fitness[0][1] < best_cost:
                best_cost = pop_fitness[0][1]
                best_path = pop_fitness[0][0]
                no_improvement = 0
            else:
                no_improvement += 1
                
            if no_improvement >= early_stopping:
                break
                
            new_pop = population[:population_size//2]
            for _ in range(population_size//2, population_size, 2):
                p1, p2 = population[np.random.randint(0, len(population)//4)], population[np.random.randint(0, len(population)//4)]
                c1, c2 = self.crossover(p1, p2)
                new_pop.extend([c1, c2])
                
            for i in range(len(new_pop)):
                if np.random.rand() < mutation_rate:
                    new_pop[i] = self.mutate(new_pop[i])
                    
            population = new_pop
            
        return best_cost, best_path, time.time()-start_time, gen+1
        
    def crossover(self, p1, p2):
        size = len(p1)
        start, end = sorted(np.random.randint(0, size, 2))
        child1, child2 = [-1]*size, [-1]*size
        child1[start:end] = p1[start:end]
        child2[start:end] = p2[start:end]
        
        for i in range(size):
            if p2[i] not in child1:
                for j in range(size):
                    if child1[j] == -1:
                        child1[j] = p2[i]
                        break
                        
            if p1[i] not in child2:
                for j in range(size):
                    if child2[j] == -1:
                        child2[j] = p1[i]
                        break
                        
        return child1, child2
        
    def mutate(self, path):
        i, j = np.random.randint(0, len(path), 2)
        path[i], path[j] = path[j], path[i]
        return path
        
    def run_simulated_annealing(self, dist_matrix, initial_temp=1000, cooling_rate=0.995, min_temp=1):
        start_time = time.time()
        n = len(dist_matrix)
        current = np.random.permutation(n)
        current_cost = sum(dist_matrix[current[i], current[(i+1)%n]] for i in range(n))
        best = current.copy()
        best_cost = current_cost
        temp = initial_temp
        iteration = 0
        
        def get_neighbor(path):
            i, j = np.random.randint(0, n, 2)
            neighbor = path.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            return neighbor
            
        while temp > min_temp:
            neighbor = get_neighbor(current)
            neighbor_cost = sum(dist_matrix[neighbor[i], neighbor[(i+1)%n]] for i in range(n))
            
            if neighbor_cost < current_cost or np.random.rand() < np.exp((current_cost-neighbor_cost)/temp):
                current, current_cost = neighbor, neighbor_cost
                
                if current_cost < best_cost:
                    best, best_cost = current.copy(), current_cost
                    
            temp *= cooling_rate
            iteration += 1
            
        return best_cost, best, time.time()-start_time, iteration
        
    def run_particle_swarm_optimization(self, dist_matrix, num_particles=50, max_iterations=1000, 
                                      inertia=0.5, cognitive=1.5, social=1.5, early_stopping=50):
        start_time = time.time()
        n = len(dist_matrix)
        particles = [np.random.permutation(n) for _ in range(num_particles)]
        velocities = [np.zeros(n) for _ in range(num_particles)]
        personal_bests = particles.copy()
        personal_costs = [float('inf')]*num_particles
        global_best = None
        global_cost = float('inf')
        no_improvement = 0
        
        def fitness(path):
            return sum(dist_matrix[path[i], path[(i+1)%n]] for i in range(n))
            
        for iteration in range(max_iterations):
            for i in range(num_particles):
                cost = fitness(particles[i])
                
                if cost < personal_costs[i]:
                    personal_costs[i] = cost
                    personal_bests[i] = particles[i].copy()
                    
                if cost < global_cost:
                    global_cost = cost
                    global_best = particles[i].copy()
                    no_improvement = 0
                else:
                    no_improvement += 1
                    
            if no_improvement >= early_stopping:
                break
                
            for i in range(num_particles):
                velocities[i] = (inertia * velocities[i] + 
                               cognitive * np.random.rand(n) * (personal_bests[i] - particles[i]) + 
                               social * np.random.rand(n) * (global_best - particles[i]))
                               
                particles[i] = np.mod(particles[i] + velocities[i], n).astype(int)
                particles[i] = np.unique(particles[i], return_index=True)[1]
                while len(particles[i]) < n:
                    missing = [x for x in range(n) if x not in particles[i]]
                    particles[i] = np.concatenate((particles[i], missing))
                    
        return global_cost, global_best, time.time()-start_time, iteration+1

def main():
    app = QApplication(sys.argv)
    solver = TSPSolverApp()
    solver.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
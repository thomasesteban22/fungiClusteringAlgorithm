#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
# Algoritmo bioinspirado en la expansión del reino fungi
class FungiInspiredClustering:
    def __init__(self,Distance_Function, n_clusters=5, max_iter=100, local_radius=1.0, global_radius=5.0, tolerance=1e-4):
        self.Distance_Function = Distance_Function
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.local_radius = local_radius
        self.global_radius = global_radius
        self.tolerance = tolerance

    def fit(self, X):
        # Inicializar las "esporas" (centroides iniciales) aleatoriamente
        np.random.seed(42)  # Para reproducibilidad
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iter):
            # Fase 1: Expansión local (caótica)
            self.labels = self.assign_local_clusters(X)

            # Fase 2: Reajuste global (dirigida)
            old_centroids = self.centroids.copy()
            self.centroids = self.calculate_new_centroids(X)

            # Verificar convergencia
            if self.has_converged(old_centroids):
                print(f"Convergencia alcanzada en la iteración {i+1}")
                break

    def assign_local_clusters(self, X):
        labels = []
        for point in X:
            distances = [self.Distance_Function(point, centroid) for centroid in self.centroids]

            # Expansión local caótica: Agregar ruido a la distancia para simular crecimiento aleatorio
            noisy_distances = [dist + np.random.uniform(-self.local_radius, self.local_radius) for dist in distances]
            labels.append(np.argmin(noisy_distances))
        return np.array(labels)

    def calculate_new_centroids(self, X):
        centroids = []
        for i in range(self.n_clusters):
            points_in_cluster = X[self.labels == i]
            if len(points_in_cluster) > 0:
                # Expansión global: Ajuste de los centroides hacia áreas ricas en puntos
                new_centroid = points_in_cluster.mean(axis=0) + np.random.uniform(-self.global_radius, self.global_radius, size=points_in_cluster.shape[1])
            else:
                # Si el clúster está vacío, reasignar un centroide aleatorio
                new_centroid = X[np.random.choice(X.shape[0])]
            centroids.append(new_centroid)
        return np.array(centroids)

    def has_converged(self, old_centroids):
        # Verificar si los centroides han cambiado por menos de la tolerancia
        distances = [self.Distance_Function(old, new) for old, new in zip(old_centroids, self.centroids)]
        return np.max(distances) < self.tolerance
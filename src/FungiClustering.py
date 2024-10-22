import numpy as np


class FungiInspiredClustering:
    """
    A bioinspired clustering algorithm based on the Fungi Kingdom Expansion (FKE) model.

    The algorithm mimics the chaotic local expansion and deterministic global adjustment
    of fungal mycelium to find optimal clusters in a dataset. It combines local noise-driven
    exploration and global centroid adjustment to cluster data points.

    Parameters
    ----------
    distance_function : callable
        A function to compute the distance between data points and centroids.
    n_clusters : int, optional (default=5)
        The number of clusters to form.
    max_iter : int, optional (default=100)
        Maximum number of iterations of the clustering algorithm.
    local_radius : float, optional (default=1.0)
        The radius of local expansion (chaotic noise) added to distances.
    global_radius : float, optional (default=5.0)
        The radius of global adjustment for centroids in each iteration.
    tolerance : float, optional (default=1e-4)
        The convergence criterion. If the change in centroids is less than this value,
        the algorithm will stop.

    Attributes
    ----------
    centroids : ndarray of shape (n_clusters, n_features)
        The final positions of the cluster centroids after fitting.
    labels : ndarray of shape (n_samples,)
        Cluster labels for each data point.
    """

    def __init__(
        self,
        distance_function,
        n_clusters=5,
        max_iter=100,
        local_radius=1.0,
        global_radius=5.0,
        tolerance=1e-4,
    ):
        """
        Initialize the FungiInspiredClustering model.

        Parameters
        ----------
        distance_function : callable
            A function to compute the distance between data points and centroids.
        n_clusters : int, optional (default=5)
            The number of clusters to form.
        max_iter : int, optional (default=100)
            Maximum number of iterations of the clustering algorithm.
        local_radius : float, optional (default=1.0)
            The radius of local expansion (chaotic noise) added to distances.
        global_radius : float, optional (default=5.0)
            The radius of global adjustment for centroids in each iteration.
        tolerance : float, optional (default=1e-4)
            The convergence criterion.
        """
        self.Distance_Function = distance_function
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.local_radius = local_radius
        self.global_radius = global_radius
        self.tolerance = tolerance

    def fit(self, X):
        """
        Fit the FungiInspiredClustering model to the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data to cluster.

        Returns
        -------
        None
        """
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
        """
        Assign each data point to the nearest cluster based on local noisy distances.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data to assign to clusters.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each data point, determined by the closest centroid.
        """
        labels = []
        for point in X:
            distances = [
                self.Distance_Function(point, centroid) for centroid in self.centroids
            ]

            # Expansión local caótica: Agregar ruido a la distancia para simular crecimiento aleatorio
            noisy_distances = [
                dist + np.random.uniform(-self.local_radius, self.local_radius)
                for dist in distances
            ]
            labels.append(np.argmin(noisy_distances))
        return np.array(labels)

    def calculate_new_centroids(self, X):
        """
        Recalculate the centroids by averaging the points in each cluster and adding global adjustment.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data to use for centroid calculation.

        Returns
        -------
        centroids : ndarray of shape (n_clusters, n_features)
            The updated positions of the cluster centroids.
        """
        centroids = []
        for i in range(self.n_clusters):
            points_in_cluster = X[self.labels == i]
            if len(points_in_cluster) > 0:
                # Expansión global: Ajuste de los centroides hacia áreas ricas en puntos
                new_centroid = points_in_cluster.mean(axis=0) + np.random.uniform(
                    -self.global_radius,
                    self.global_radius,
                    size=points_in_cluster.shape[1],
                )
            else:
                # Si el clúster está vacío, reasignar un centroide aleatorio
                new_centroid = X[np.random.choice(X.shape[0])]
            centroids.append(new_centroid)
        return np.array(centroids)

    def has_converged(self, old_centroids):
        """
        Check if the centroids have converged, based on the tolerance threshold.

        Parameters
        ----------
        old_centroids : ndarray of shape (n_clusters, n_features)
            The previous centroid positions.

        Returns
        -------
        bool
            True if the centroids have moved less than the tolerance, otherwise False.
        """
        distances = [
            self.Distance_Function(old, new)
            for old, new in zip(old_centroids, self.centroids)
        ]
        return np.max(distances) < self.tolerance

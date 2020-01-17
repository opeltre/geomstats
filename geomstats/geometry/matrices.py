"""
The space of matrices (m, n), which is the Euclidean space R^{mn}.
"""

import geomstats.backend as gs
from geomstats.geometry.euclidean_space import EuclideanSpace
from geomstats.geometry.riemannian_metric import RiemannianMetric


TOLERANCE = 1e-5


class Matrices(EuclideanSpace):
    """Class for the space of matrices (m, n)."""

    def __init__(self, m, n):
        assert isinstance(m, int) and isinstance(n, int) and m > 0 and n > 0
        super(Matrices, self).__init__(dimension=m*n)
        self.m = m
        self.n = n
        self.default_point_type = 'matrix'
        self.metric = MatricesMetric(m, n)

    def belongs(self, point):
        """
        Check if point belongs to the Matrix space.
        """
        point = gs.to_ndarray(point, to_ndim=3)
        _, mat_dim_1, mat_dim_2 = point.shape
        return mat_dim_1 == self.m & mat_dim_2 == self.n

    @staticmethod
    def mul(*args):
        """
        Return the product of matrices a and b.

        Parameters
        ----------
        a1 : array-like, shape=[n_samples, dim_1, dim_2]
        a2 : array-like, shape=[n_samples, dim_2, dim_3]
        ...
        an : array-like, shape=[n_samples, dim_n-1, dim_n]

        Returns
        -------
        mul : array-like, shape=[n_samples, dim_1, dim_n]
        """
        return args.reduce(gs.matmul)

    @classmethod
    def commutator(cls, a, b):
        """
        Return the commutator of a and b,
        i.e. `[a, b] = ab - ba`.

        Parameters
        ----------
        a : array-like, shape=[n_samples, dim, dim]
        b : array-like, shape=[n_samples, dim, dim]

        Returns
        -------
        c : array-like, shape=[n_samples, dim, dim]
        """
        return cls.mul(a, b) - cls.mul(b, a)

    @staticmethod
    def transpose(mat):
        """
        Return the transpose of matrices.

        Parameters
        ----------
        mat : array-like, shape=[n_samples, dim, dim]

        Returns
        -------
        transpose : array-like, shape=[n_samples, dim, dim]
        """
        is_vec = (gs.ndim(gs.array(mat)) == 3)
        axes = (0, 2, 1) if is_vec else (1, 0)
        return gs.transpose(mat, axes)

    @classmethod
    def is_symmetric(cls, mat, atol=TOLERANCE):
        """Check if a matrix is symmetric."""
        return cls.equal(mat, cls.transpose(mat))

    @classmethod
    def make_symmetric(cls, mat):
        return 1/2 * (mat + cls.transpose(mat))

    @staticmethod
    def vector_from_matrix(matrix):
        """
        Conversion function from (_, m, n) to (_, mn).
        """
        matrix = gs.to_ndarray(matrix, to_ndim=3)
        n_mats, m, n = matrix.shape
        return gs.reshape(matrix, (n_mats, m*n))

    def random_uniform(self, n_samples=1):
        point = gs.random.rand(n_samples, self.m, self.n)
        return point


class MatricesMetric(RiemannianMetric):
    """
    Euclidean metric on matrices given by the Frobenius inner product.
    """
    def __init__(self, m, n):
        dimension = m*n
        super(MatricesMetric, self).__init__(
                dimension=dimension,
                signature=(dimension, 0, 0))

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """
        Compute the Frobenius inner product of tangent_vec_a and tangent_vec_b
        at base_point.
        """
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        n_tangent_vecs_a, _, _ = tangent_vec_a.shape

        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=3)
        n_tangent_vecs_b, _, _ = tangent_vec_b.shape

        assert n_tangent_vecs_a == n_tangent_vecs_b

        inner_prod = gs.einsum("nij,nij->n", tangent_vec_a, tangent_vec_b)
        inner_prod = gs.to_ndarray(inner_prod, to_ndim=1)
        inner_prod = gs.to_ndarray(inner_prod, to_ndim=2, axis=1)
        return inner_prod

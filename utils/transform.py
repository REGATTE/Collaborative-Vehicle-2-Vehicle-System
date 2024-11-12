import numpy as np
import math


# in yaw pitch roll sequence
class Transform(object):
    def __init__(self, x, y, z, yaw, pitch, roll):
        self._x = x  # in meters
        self._y = y
        self._z = z
        self._yaw = yaw  # in degrees
        self._pitch = pitch
        self._roll = roll
        cy = math.cos(np.radians(yaw))
        sy = math.sin(np.radians(yaw))
        cr = math.cos(np.radians(roll))
        sr = math.sin(np.radians(roll))
        cp = math.cos(np.radians(pitch))
        sp = math.sin(np.radians(pitch))
        self._matrix = np.identity(4)
        self._matrix[0, 3] = x
        self._matrix[1, 3] = y
        self._matrix[2, 3] = z
        self._matrix[0, 0] = (cp * cy)
        self._matrix[0, 1] = (cy * sp * sr - sy * cr)
        self._matrix[0, 2] = -(cy * sp * cr + sy * sr)
        self._matrix[1, 0] = (sy * cp)
        self._matrix[1, 1] = (sy * sp * sr + cy * cr)
        self._matrix[1, 2] = (cy * sr - sy * sp * cr)
        self._matrix[2, 0] = (sp)
        self._matrix[2, 1] = -(cp * sr)
        self._matrix[2, 2] = (cp * cr)
        self._rcw = self._matrix[:3, :3]
        self._tcw = self._matrix[:3, 3]
        """inverse transform"""
        self._inversematrix = np.linalg.inv(self._matrix)

    @staticmethod
    def transform_with_matrix(points, transform_matrix):
        """
            points: (n, 3) numpy array of [x, y, z]
            transform_matrix: (4, 4) numpy array
        """
        points = points.T
        points = np.append(points, np.ones((1, points.shape[1])), axis=0)
        transformed_points = transform_matrix @ points
        return transformed_points[:3].T

    def transform(self, points):
        """
            points: (n, 3) numpy array of [x, y, z]
        """
        return Transform.transform_with_matrix(points, self._matrix)

    def inverse_transform(self, points):
        """
            points: (n, 3) numpy array of [x, y, z]
        """
        return Transform.transform_with_matrix(points, self._inversematrix)

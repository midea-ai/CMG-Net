import numpy as np
import math
import torch

# axis
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def normal_vector(vector):
    vector_norm = np.linalg.norm(vector, axis=1)
    vector = (vector / vector_norm)
    return vector


def euclideanDistance(instance1, instance2):
    instance1_copy = instance1.reshape((3, -1))
    instance2_copy = instance2.reshape((3, -1))
    distance = 0
    for i in range(3):
        distance += (instance1_copy[i] - instance2_copy[i]) ** 2

    return math.sqrt(distance)


def rad2deg(rads):
    return 180. * rads / math.pi


def deg2rad(degs):
    return math.pi * degs / 180.


def get_distance_vertices_batched(obj, hand):
    n1 = len(hand[1])
    n2 = len(obj[0])

    matrix1 = hand.unsqueeze(1).repeat(1, n2, 1, 1)
    matrix2 = obj.unsqueeze(2).repeat(1, 1, n1, 1)
    dists = torch.norm(matrix1 - matrix2, dim=-1)
    dists = dists.min(1)[0]

    return dists


def get_normal_face_batched(vertices, faces):
    p1 = vertices[faces[:, 0]]
    p2 = vertices[faces[:, 1]]
    p3 = vertices[faces[:, 2]]
    U = p2 - p1
    V = p3 - p1
    Nx = U[:, 1] * V[:, 2] - U[:, 2] * V[:, 1]
    Ny = U[:, 2] * V[:, 0] - U[:, 0] * V[:, 2]
    Nz = U[:, 0] * V[:, 1] - U[:, 1] * V[:, 0]
    return -1 * torch.stack((Nx, Ny, Nz)).T

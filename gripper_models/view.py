import trimesh
import numpy as np
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from transformer import Transform

mesh_name = './barrett_hand/finger_tip.stl'
mesh = trimesh.load(mesh_name)
mesh.show()
mesh_name_1 = './barrett_hand/palm_280.stl'
mesh_1 = trimesh.load(mesh_name_1)
mesh_1.show()

finger_tip_vertices = [35, 98, 44, 90, 95, 77, 84, 73, 76, 25, 108, 64, 22, 24, 96, 23, 85, 79, 83, 30, 45, 47, 68, 54,
                       42, 69,
                       92, 86, 19, 7, 94, 37, 99, 91, 11, 107, 0, 89, 57, 59, 109, 4, 65, 31, 2, 1, 10, 101, 52, 97, 87,
                       50, 72, 15, 106, 82, 12, 56, 78, 32, 46, 8]
points_visual = trimesh.points.PointCloud(mesh.vertices[finger_tip_vertices])

point = np.asarray([0.0362, -0.0286, 0.0005])
r = 0.012

pose = Transform.identity()
pose.translation += point
pose_matrix = pose.as_matrix()
axis = trimesh.creation.axis(origin_size=0.002, transform = pose_matrix)
# axis = trimesh.creation.axis(0.002)

circle = trimesh.primitives.Sphere(radius=r, center=point)

scene = trimesh.Scene([
    circle,
    mesh,
    axis
])

scene.show(smooth=False)


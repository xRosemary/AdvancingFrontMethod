import numpy as np
import alphashape
from shapely.geometry import mapping
import PharseOBJfile

class Separator():

    @staticmethod
    def get_alpha(mesh_vertex, radius=0.1, default_value=-1):
        boundary = []
        points = [(i, j) for i, j in zip(mesh_vertex[:, 0], mesh_vertex[:, 1])]
        alpha_shape = alphashape.alphashape(points, radius)
        data = mapping(alpha_shape)
        for i in data['coordinates']:
            for j in i:
                boundary.append(list(j))

        inside = [x for x in boundary if x not in points]
        np.array(boundary)
        np.array(inside)

        boundary = np.insert(boundary, 1, default_value, axis=1)
        inside = np.insert(inside, 1, default_value, axis=1)
        # 倒转boundary的顺序
        boundary = boundary[::-1]
        return boundary, inside


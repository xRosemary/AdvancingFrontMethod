from ctypes import pointer
import numpy as np
import alphashape
from shapely.geometry import mapping
import PharseOBJfile
import matplotlib.pyplot as plt

class Separator():

    @staticmethod
    def get_alpha(mesh_vertex, radius=0.1, default_value=-1):
        boundary = []
        points = [(i, j) for i, j in zip(mesh_vertex[:, 0], mesh_vertex[:, 1])]
        points = np.array(points)
        # 找出最短距离
        min_distance = float('inf')
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = np.sqrt(np.sum((points[i] - points[j]) ** 2))
                if distance < min_distance:
                    min_distance = distance

        alpha_shape = alphashape.alphashape(points, radius)
        data = mapping(alpha_shape)
        for i in data['coordinates']:
            for j in i:
                boundary.append(list(j))

        inside = []
        for p in points:
            isBoundary = False

            for b in boundary:
                
                if p[0] == b[0] and p[1] == b[1]:
                    isBoundary = True
                    break

            if not isBoundary:
                inside.append(p)
        
        np.array(boundary)
        np.array(inside)

        boundary = np.insert(boundary, 1, default_value, axis=1)
        inside = np.insert(inside, 1, default_value, axis=1)
        # 倒转boundary的顺序
        boundary = boundary[::-1]
        # 删除boundary的最后一个点
        boundary = boundary[:-1]
        return boundary, inside, min_distance

# main
if __name__ == '__main__':
    path = "input3.obj"
    vertex, objectNorm, mesh = PharseOBJfile.read2dObjShap(path)
    boundary, inside, min_distance = Separator.get_alpha(vertex)
    # print(boundary, inside)

    print(boundary)
    print(inside)

    plt.scatter(boundary[:, 0], boundary[:, 2], c='r', marker='o')
    plt.scatter(inside[:, 0], inside[:, 2], c='b', marker='o')
    plt.show()
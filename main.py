import matplotlib.pyplot as plt
import numpy as np
import turtle
import PharseOBJfile
from Alphashape import Separator

class AdvancingFrontMesh():
    def __init__(self):
        self.nmax = 10000
        # 0行3列的点集
        self.point = np.empty((0, 3))
        # 前向点下标
        self.PointFront = np.array([0 for _ in range(self.nmax)], dtype=int)
        # 三角形顶点
        self.verties = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        # 三角形索引
        self.tries = []

        # 用于链式前向星的变量

        self.EdgeFrom = np.array([0 for _ in range(self.nmax)], dtype=int)  # 边的起点
        self.EdgeTo = np.array([0 for _ in range(self.nmax)], dtype=int)  # 边的终点
        self.EdgeNext = np.array([0 for _ in range(self.nmax)], dtype=int)  # 相同起点的上一条边
        self.Head = np.array([-1 for _ in range(self.nmax)], dtype=int)  # 方括号里填起点编号，可得到起点最新的边
        self.Edgenum = 0  # 总边数

        # 波前推进法的一些变量
        self.EdgeFront = np.array([False for _ in range(self.nmax)], dtype=bool) # 当前边是否为前边(front)
        self.EdgeIndex = 0  # 目前索引数
        self.Triangles = np.array([], dtype=int) # 三角形索引
        self.TriangleRadius = 0.0  # 等边三角形的边长，(1-0.5^2)^(1/2) = 0.866
        self.height = 0.0  # 从边的中点生出新点的距离，尽量保持为等边三角形

    def split_point(self, path, size=50):
        # 分割点集，将模型的点集分割成边界点和内部点
        # path: 模型的路径
        # size: 绘图时放大的倍数

        vertex, objectNorm, mesh = PharseOBJfile.read2dObjShap(path)
        boundary, inside, min_distance = Separator.get_alpha(vertex)

        self.TriangleRadius = min_distance / 2  # 圆形半径
        self.height = self.TriangleRadius  # 三角形的高

        self.point = boundary
        for i in range(len(boundary)-1):
            self.add_edge(i, i+1)

        self.add_edge(len(boundary)-1, 0)
        self.point = np.append(self.point, inside, axis=0)

        self.point = np.append(self.point, [self.point[0]], axis=0)
        print(self.point)
        for i in range(len(self.point)):
            self.PointFront[i] = 2

        points = np.delete(self.point,1, axis = 1) * size

        # 绘制已有的点
        for p in points:
            turtle.penup()
            turtle.goto(p)
            turtle.dot()
            turtle.pendown()
 
    def get_new_triangle(self):
        # 根据前向点和边的索引，生成新的三角形

        st = self.EdgeFrom[self.EdgeIndex]
        ed = self.EdgeTo[self.EdgeIndex]

        i = self.Head[ed]
        while(i != -1):
            if(self.EdgeFront[i] == False):
                i = self.EdgeNext[i]
                continue
            
            to = self.EdgeTo[i]

            j = self.Head[to]
            while(j != -1):
                if(self.EdgeFront[j] == False):
                    j = self.EdgeNext[j]
                    continue

                if(self.EdgeTo[j] == st):
                    self.Triangles = np.append(self.Triangles, st)
                    self.Triangles = np.append(self.Triangles, to)
                    self.Triangles = np.append(self.Triangles, ed)

                    self.PointFront[st] -= 2
                    self.PointFront[to] -= 2
                    self.PointFront[ed] -= 2
                    self.EdgeFront[i] = False
                    self.EdgeFront[j] = False
                    self.EdgeFront[self.EdgeIndex] = False
                    self.EdgeIndex += 1
                    return
                
                j = self.EdgeNext[j]

            i = self.EdgeNext[i]

        StartVec = self.point[st]
        EndVec = self.point[ed]

        MidVec = (StartVec + EndVec) / 2

        Normal = np.array([-(EndVec[2] - StartVec[2]), 0.0, EndVec[0] - StartVec[0]])

        Normal = Normal / np.linalg.norm(Normal)

        NewVec = MidVec + Normal * self.height  # 新点的位置
        newindex = -1
        mindis = 9999.0

        leftedge = True
        rightedge = True

        for i in range(len(self.point)):
            if(i == st or i == ed):
                continue
            if(self.PointFront[i] == 0):
                continue
            
            dis = np.linalg.norm(NewVec - self.point[i])
            if(dis < self.TriangleRadius and dis < mindis):
                mindis = dis
                newindex = i       

        if(newindex == -1):
            newindex = len(self.point)
            self.point = np.append(self.point, [NewVec], axis=0)
            self.PointFront[newindex] += 2
        else:
            i = self.Head[newindex]
            while(i != -1):
                if (self.EdgeTo[i] == st):
                    self.EdgeFront[i] = False
                    leftedge = False;  # 无需再新建三角形的左边
                    self.PointFront[st] -= 2
                    break
                i = self.EdgeNext[i]

            if(leftedge == True):
                i = self.Head[ed]
                while(i != -1):
                    if(self.EdgeTo[i] == newindex):
                        self.EdgeFront[i] = False
                        rightedge = False
                        self.PointFront[newindex] -= 2
                        break
                    i = self.EdgeNext[i]
            if(leftedge == True and rightedge == True):
                self.PointFront[newindex] += 2

        
        self.EdgeFront[self.EdgeIndex] = False

        self.Triangles = np.append(self.Triangles, st)
        self.Triangles = np.append(self.Triangles, newindex)
        self.Triangles = np.append(self.Triangles, ed)

        if (leftedge):
            self.add_edge(st, newindex)
        if (rightedge):
            self.add_edge(newindex, ed)

        self.EdgeIndex += 1
        
    def Update(self):
        # 更新点的前向点和边的前向点
        
        if (self.EdgeIndex < self.Edgenum):
        
            if (self.EdgeFront[self.EdgeIndex] == True):
            
                self.get_new_triangle()
                self.verties = np.empty(shape=(len(self.Triangles),3), dtype=int)
                self.tries = np.empty(shape=(len(self.Triangles),0), dtype=int)

                for i in range(len(self.Triangles)):
                    self.verties[i] = np.array(self.point[self.Triangles[i]])
                    self.tries[i] = i
            else:
                self.EdgeIndex += 1

            return True
        else:
            return False

    def add_edge(self, st, ed):
        # 添加边
        # st: 边的起点
        # ed: 边的终点
        self.EdgeFrom[self.Edgenum] = st
        self.EdgeTo[self.Edgenum] = ed
        self.EdgeNext[self.Edgenum] = self.Head[st]
        self.Head[st] = self.Edgenum
        self.EdgeFront[self.Edgenum] = True
        self.Edgenum += 1

    def Draw(self, size=50):
        # 绘制三角形
        # size: 绘制时放大的倍数

        self.Triangles = self.Triangles.reshape(-1,3)
        turtle.color("blue")
        for tri in self.Triangles:
            # 设置画笔的起点为points[0]
            turtle.penup()
            points = [self.point[tri[0]], self.point[tri[1]], self.point[tri[2]]]
            points = np.delete(points,1, axis = 1) * size
            turtle.goto(points[0])
            turtle.pendown()
            #turtle.begin_fill()
            turtle.goto(points[1])  # 走到points[1]
            turtle.goto(points[2])  # 走到points[2]
            turtle.goto(points[0])  # 回到points[0]，完成三角形的绘制
    
        turtle.done()


if __name__ == '__main__':
    path = "input3.obj"
    
    AFM = AdvancingFrontMesh()

    size = 120

    AFM.split_point(path,size=size)

    while(AFM.Update()):
        pass
    
    print("Done")
    AFM.Draw(size=size)
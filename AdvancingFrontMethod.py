from sqlalchemy import null
from Structure import *
from unityMethod import PharseOBJfile
import matplotlib.pyplot as plt
import numpy as np
import alphashape
from shapely.geometry import mapping
import turtle


class AdvancingFrontMethod():
    def __init__(self, v, objectNorm, max_num):
        # # 模型的相关数据
        # self.vertex = 100 * v
        self.max_num = max_num
        # # 找出模型的边界点,和边界内的点
        # self.boundary, self.inside = self.get_alpha()

        
        self.Edge = []
        self.EdgeFront = [False for _ in range(self.max_num)]
        self.pointFront = [2 for _ in range(self.max_num)]

        self.tries = []

        self.EdgeFrom = [0 for _ in range(self.max_num)]
        self.EdgeTo = [0 for _ in range(self.max_num)]
        self.EdgeNext = [0 for _ in range(self.max_num)]
        self.Head = [-1 for _ in range(self.max_num)]
        self.EdgeNum = 0
        self.EdgeIndex = 0
        self.verties = [0,0,0]
        self.Triangles = [] 
        self.point = []
        self.height = -30
        self.TriangleRadius = 30


        self.init1()
        self.point = np.array(self.point)

        # for i in range(len(self.boundary)):
        #     self.Edge.append(
        #         (self.boundary[i], self.boundary[(i+1)%len(self.boundary)])
        #     )
        
        # for i in range(len(self.boundary)):
        #     self.addEdge(i,i+1)

        # 获取边界坐标的最值
        # self.x_max = max([p[0] for p in self.boundary])
        # self.x_min = min([p[0] for p in self.boundary])
        # self.y_max = max([p[1] for p in self.boundary])
        # self.y_min = min([p[1] for p in self.boundary])
        # print(self.x_max, self.x_min, self.y_max, self.y_min)
        

    def get_Triangle(self):

        if (self.EdgeIndex < self.EdgeNum):
        
            if (self.EdgeFront[self.EdgeIndex] == True):
            
                self.GenNewTriangle()
                self.verties = [[0,0,0] for _ in range(len(self.Triangles))]
                self.tries = [0 for _ in range(len(self.Triangles))]

                for i in range(len(self.Triangles)):
                    self.verties[i] = self.point[self.Triangles[i]]
                    self.tries[i] = i
            
            else:
                self.EdgeIndex += 1
                
    def init1(self):
        
        self.TriangleRadius = 0.9
        self.height = self.TriangleRadius * 0.9
        squarelen = 10
        distan = self.TriangleRadius

        for i in range(squarelen):
            self.point.append([distan*i, 0])
            self.addEdge(i, i+1)

        for i in range(squarelen):
            self.point.append([squarelen * distan, i * distan])
            self.addEdge(i + squarelen, i + 1 + squarelen)

        for i in range(squarelen):
            self.point.append([squarelen * distan - i * distan, squarelen * distan])
            self.addEdge(i + 2 * squarelen, i + 1 + 2 * squarelen)

        for i in range(squarelen):
            self.point.append([0, squarelen * distan - i * distan])
            ed = i + 1 + squarelen * 3
            if (i == squarelen - 1):
                ed = 0

            self.addEdge(i + squarelen * 3, ed)

          
        CircleCenter = [3.0, 3.0]
        CircleRadius = 2.0
        CircleParticle = 16
        for i in range(CircleParticle):
            self.point.append([CircleCenter[0] + CircleRadius * np.cos(2 * np.pi * i / CircleParticle), CircleCenter[1] + CircleRadius * np.sin(2 * np.pi * i / CircleParticle)])
            ed = i + 1 + squarelen * 4
            if (i == CircleParticle - 1):
                ed = 4 * squarelen
            self.addEdge(i + squarelen * 4, ed)



    def GenNewTriangle(self):
        st = self.EdgeFrom[self.EdgeIndex]
        ed = self.EdgeTo[self.EdgeIndex]

        i = self.Head[ed]

        while(i != -1):
            
            if(self.EdgeFront[i] == False):
                i = self.EdgeNext[i]
                continue
            to = self.EdgeTo[i]
            i = self.EdgeNext[i]

            j = self.Head[to]
            while(j != -1):
                
                if(self.EdgeFront[j] == False):
                    j = self.EdgeNext[j]
                    continue
                if(self.EdgeTo[j] == st):
                    self.Triangles.append(st)
                    self.Triangles.append(to)
                    self.Triangles.append(ed)
                    self.pointFront[st] -= 2
                    self.pointFront[to] -= 2
                    self.pointFront[ed] -= 2
                    self.EdgeFront[i] = False
                    self.EdgeFront[j] = False
                    self.EdgeFront[self.EdgeIndex] = False
                    self.EdgeIndex += 1
                    return
                j = self.EdgeNext[j]

            

        StartVec = self.point[st]
        EndVec = self.point[ed]

        midpoint = (StartVec + EndVec) / 2
        normal = np.array([-EndVec[1] + StartVec[1], EndVec[0] - StartVec[0]])
        
        unitVector = normal / np.linalg.norm(normal)

        NewVec = midpoint + unitVector * self.height

        newindex = -1

        mindis = 9999.0
        
        leftedge = True
        rightedge = True

        for i in range(len(self.point)):
            if(i == st or i == ed):
                continue
            if(self.pointFront[i] == 0):
                continue

            dis = np.linalg.norm(self.point[i] - NewVec)
            if(dis < self.TriangleRadius and dis < mindis):
                mindis = dis
                newindex = i
            
        if (newindex == -1):
            newindex = len(self.point)
            self.point = np.append(self.point, [NewVec], axis=0)
            
            self.pointFront[newindex] += 2
        else:
            i = self.Head[newindex]

            while(i != -1):
                if(self.EdgeTo[i] == st):
                    self.EdgeFront[i] = False
                    leftedge = False
                    self.pointFront[st] -= 2
                    break
                if(leftedge == True):
                    j = self.Head[ed]
                    while (j != -1):
                        if(self.EdgeTo[j] == newindex):
                            self.EdgeFront[j] = False
                            self.pointFront[newindex] -= 2
                            rightedge = False
                            break
                        j = self.EdgeNext[j]
                if (leftedge == True and rightedge == True):
                    self.pointFront[newindex] += 2

                i = self.EdgeNext[i]

        for i in range(len(self.point)):
            if(i == st or i == ed):
                continue
            if(self.pointFront[i] == 0):
                continue
            dis = np.linalg.norm(self.point[i] - NewVec)
            if(dis < self.TriangleRadius and dis < mindis):
                mindis = dis
                newindex = i
        if(newindex == -1):
            newindex = len(self.point)
            self.point.append(NewVec)
            self.pointFront[newindex] += 2
        else:
            i = self.Head[newindex]
            while(i != -1):
                if(self.EdgeTo[i] == st):
                    self.EdgeFront[i] = False
                    leftedge = False
                    self.pointFront[st] -= 2
                    break
                i = self.EdgeNext[i]

            if(leftedge == True):
                j = self.Head[ed]
                while (j != -1):
                    if(self.EdgeTo[j] == newindex):
                        self.EdgeFront[j] = False
                        self.pointFront[newindex] -= 2
                        rightedge = False
                        break
                    j = self.EdgeNext[j]

            if (leftedge == True and rightedge == True):
                self.pointFront[newindex] += 2


        self.EdgeFront[self.EdgeIndex] = False
        self.Triangles.append(st)
        self.Triangles.append(newindex)
        self.Triangles.append(ed)

        if (leftedge):
            self.addEdge(st, newindex)
        if (rightedge):
            self.addEdge(newindex, ed)
        self.EdgeIndex += 1




    def addEdge(self, st, ed):
        self.EdgeFrom[self.EdgeNum] = st
        self.EdgeTo[self.EdgeNum] = ed
        self.EdgeNext[self.EdgeNum] = self.Head[st]
        self.Head[st] = self.EdgeNum
        self.EdgeFront[self.EdgeNum] = True
        self.EdgeNum += 1

    def getTriangles(self, height, radius=1):
        triangles = []
        points = []

        while(self.Edge):

            edge = self.Edge.pop(0)
            
            vertex1 = edge[0]
            vertex2 = edge[1]

            # 获取两点的中垂线
            midpoint = (vertex1 + vertex2) / 2
            normal = np.array([-vertex2[1] + vertex1[1], vertex2[0] - vertex1[0]])
            
            unitVector = normal / np.linalg.norm(normal)

            newPoint = midpoint + unitVector * height

            
            getPoint = False

            

            for v in self.inside:
                # 判断点v与newPoint的距离是否小于1
                if np.linalg.norm(v - newPoint) < radius:
                    getPoint = True
                    triangles.append([vertex1, vertex2, v])
                    self.Edge.append((vertex1, v))
                    self.Edge.append((v, vertex2))

                    points.append([v[0], v[1], 0])

                    # if len(self.inside) <= 2:
                    #     return triangles, points


                    # self.inside = np.delete(self.inside, np.where(self.inside == v), axis=0)
                    break
            
            if not getPoint:

                self.inside = np.append(self.inside, [newPoint], axis=0)

                triangles.append([vertex1, vertex2, newPoint])
                self.Edge.append((vertex1, newPoint))
                self.Edge.append((newPoint, vertex2))

                points.append([newPoint[0], newPoint[1], 0])

        return triangles, points


    def drawTriangles(self):
        # triangles, points = self.getTriangles(height=-5, radius=10)
        # points = np.array(points)

        # points = np.append(points, self.vertex, axis=0)
        # points += 1
        # colors = ["blue", "red"]
            

        # for i in range(len(points)):
        #     plt.plot(points[i, 0], points[i, 1], color=colors[int(points[i, 2])], marker="o", markersize=7, ls="None")

        # plt.show()

        # for tri in triangles:
        
        triangles = np.array(self.Triangles)
        triangles = triangles.reshape(-1,3)

        for tri in triangles:
            # plt.plot(self.point[tri[0]][0], self.point[tri[0]][1], color="red", marker="o", markersize=7, ls="None")
            # plt.plot(self.point[tri[1]][0], self.point[tri[1]][1], color="blue", marker="o", markersize=7, ls="None")
            # plt.plot(self.point[tri[2]][0], self.point[tri[2]][1], color="green", marker="o", markersize=7, ls="None")
            # overRange = False
            # for i in tri:
            #     if(self.point[i][0] < self.x_min or self.point[i][0] > self.x_max or self.point[i][1] < self.y_min or self.point[i][1] > self.y_max):
            #         overRange = True
            #         continue
            # if(overRange):
            #     continue
            self.drawTriangle(tri=tri)

        plt.show()


 
    def drawTriangle(self, tri):
    
        turtle.color("blue")
    
        # 设置画笔的起点为points[0]
    
        turtle.penup()
        points = [self.point[tri[0]], self.point[tri[1]], self.point[tri[2]]]

        turtle.goto(20 * points[0])
    
        turtle.pendown()
    
        #turtle.begin_fill()
    
        turtle.goto(20 * points[1])  # 走到points[1]
    
        turtle.goto(20 * points[2])  # 走到points[2]
    
        turtle.goto(20 * points[0])  # 回到points[0]，完成三角形的绘制
    
        #turtle.end_fill()
        
    
    
    def DrawVetex(self):
        plt.scatter(self.vertex[:,0], self.vertex[:,1])

    def ScatterWithBoundries(self):
        plt.scatter(self.boundary[:,0], self.boundary[:,1])
        plt.plot(self.boundary[:,0], self.boundary[:,1], 'k-', linewidth=1.5)

    def get_alpha(self, radius=0.1):   
        boundary = []
        points = [(i,j) for i, j in zip(self.vertex[:,0],self.vertex[:,1])]
        alpha_shape = alphashape.alphashape(points,radius)   
        data = mapping(alpha_shape)    
        for i in data['coordinates']:
            for j in i:
                boundary.append(list(j))
        
        inside = [x for x in boundary if x not in points]

        return np.array(boundary), np.array(inside)


if __name__ == '__main__':
    vertex, objectNorm, mesh = PharseOBJfile.read2dObjShap("D:\\MyCodeProject\\vsCodeProject\\cppFile\\图形学\\AdvancingFrontMethod\\input3.obj")
    AFM = AdvancingFrontMethod(vertex, objectNorm, 9999)
    for i in range(500):
        AFM.get_Triangle()
        #print(AFM.Triangles)
    print("Done")
    AFM.drawTriangles()
    
    # AFM.DrawVetex()
    # plt.show()
    # Triangles' index are the AFM.Triangles
    # 根据Triangles的index绘制三角形



    # tri = np.array(AFM.Triangles)
    # tri = tri.reshape(-1,3)
    # print(tri)
    # AFM.ScatterWithBoundries()
    # AFM.DrawVetex()
    # plt.show()
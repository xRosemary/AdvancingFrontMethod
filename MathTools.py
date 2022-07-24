# myOwnMathTools
import numpy as np
import sys

def dot(a,b):
    return np.dot(a,np.transpose(b))

def norm2d(a):
    return np.sqrt(a[0]**2+a[1]**2)

def norm(a):    
    return np.sqrt(a[0]**2+a[1]**2+a[2]**2)

def remove(duplicate): 
    final_list = [] 
    for num in duplicate: 
        if num not in final_list: 
            final_list.append(num) 
    return final_list 

def removeRepeatingVectorInList(vector3List,errorDecimal):
    final_list = []
    finalStrList = []
    for vector3 in vector3List:
        strtemp = str(round(vector3[0],errorDecimal))+str(round(vector3[1],errorDecimal))
        if strtemp not in finalStrList: 
            finalStrList.append(strtemp)
            final_list.append(vector3)
    final_list = np.array(final_list)
    return final_list 


def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b)
    common = False
    commonElements = a_set.intersection(b_set)
    if len(commonElements) > 0: 
        common = True  
    return common,list(commonElements)

def pointProjectionOnTriangle(p, triangle_vertex_0, triangle_vertex_1, triangle_vertex_2):
    # -------------------------------------------------------------------
    # Determine if projection of 3D point onto plane is within a triangle
    # https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle
    # -------------------------------------------------------------------
    # u=P2−P1
    u = triangle_vertex_1 - triangle_vertex_0
    # v=P3−P1
    v = triangle_vertex_2 - triangle_vertex_0
    # n=u×v
    n = np.cross(u,v)
    # w=P−P1
    w = p - triangle_vertex_0
    # Barycentric coordinates of the projection P′of P onto T:
    # γ=[(u×w)⋅n]/n²
    nsquare = np.dot(n,np.transpose(n))
    gamma = float(np.dot(np.cross(u,w),np.transpose(n))/nsquare)
    # β=[(w×v)⋅n]/n²
    beta = float(np.dot(np.cross(w,v),np.transpose(n))/nsquare)
    alpha = 1 - gamma - beta
    # The point P′ lies inside T if:
    # ((0 <= alpha) && (alpha <= 1) && (0 <= beta)  && (beta  <= 1) && (0 <= gamma) && (gamma <= 1))
    inside = False
    epsilon = 0.00000000 # introduce the error part to make the estimation more conservertive
    if (epsilon <= alpha) and (alpha <= 1-epsilon) and (epsilon <= beta)  and (beta  <= 1-epsilon) and ( epsilon<= gamma) and (gamma <= 1-epsilon):
        inside = True
    #if (0 <= alpha) and (alpha <= 1) and (0 <= beta)  and (beta  <= 1) and (0 <= gamma) and (gamma <= 1):
        #inside = True
    # P′= αP1+βP2+γP3
    pPrime = alpha*triangle_vertex_0 + beta*triangle_vertex_1 + gamma*triangle_vertex_2
    return inside,pPrime

def pointInsideTriangle(p, triangle_vertex_0, triangle_vertex_1, triangle_vertex_2):
    # -------------------------------------------------------------------
    # Determine if projection of 3D point onto plane is within a triangle
    # https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle
    # -------------------------------------------------------------------
    # u=P2−P1
    u = triangle_vertex_1 - triangle_vertex_0
    # v=P3−P1
    v = triangle_vertex_2 - triangle_vertex_0
    # n=u×v
    n = np.cross(u,v)
    # w=P−P1
    w = p - triangle_vertex_0
    # Barycentric coordinates of the projection P′of P onto T:
    # γ=[(u×w)⋅n]/n²
    nsquare = np.dot(n,np.transpose(n))
    gamma = float(np.dot(np.cross(u,w),np.transpose(n))/nsquare)
    # β=[(w×v)⋅n]/n²
    beta = float(np.dot(np.cross(w,v),np.transpose(n))/nsquare)
    alpha = 1 - gamma - beta
    # The point P′ lies inside T if:
    # ((0 <= alpha) && (alpha <= 1) && (0 <= beta)  && (beta  <= 1) && (0 <= gamma) && (gamma <= 1))
    inside = False
    epsilon = 0.000000001 # introduce the error part to make the estimation more conservertive
    if (-epsilon <= alpha ) and (alpha <= 1 + epsilon) and (-epsilon <= beta )  and (beta  <= 1 + epsilon) and ( -epsilon <= gamma ) and (gamma <= 1 + epsilon):
        inside = True
    #if (0 <= alpha) and (alpha <= 1) and (0 <= beta)  and (beta  <= 1) and (0 <= gamma) and (gamma <= 1):
        #inside = True
    # P′= αP1+βP2+γP3
    #pPrime = alpha*triangle_vertex_0 + beta*triangle_vertex_1 + gamma*triangle_vertex_2
    return inside,[alpha,beta,gamma]

def pointToLineSegmentDistance(v,a,b):
    ab  = b - a
    av  = v - a 

    if np.dot(av,np.transpose(ab)) <= 0.0: # Point is lagging behind start of the segment, so perpendicular distance is not viable.
        return np.inf,np.array([0,0,0])
    bv  = v - b
    if np.dot(bv,np.transpose(ab)) >= 0.0: # Point is advanced past the end of the segment, so perpendicular distance is not viable.
        return np.inf,np.array([0,0,0])       
    np.max 
    abNorm = np.sqrt(ab[0]**2+ab[1]**2+ab[2]**2)
    
    abCrossAv = np.cross(ab,av)
    abCrossAvNorm = np.sqrt(abCrossAv[0]**2+abCrossAv[1]**2+abCrossAv[2]**2)
    distance = abCrossAvNorm/abNorm

    abUnit = ab/abNorm

    pPrime = a + np.dot(abUnit,np.transpose(av))*abUnit
    return distance, pPrime     # distance: Perpendicular distance of point to segment. pPrime: projection point on the line segment

def dstFromVertexToTriangle(vertex,tri1,tri2,tri3):
    inside,tempPPrime = pointProjectionOnTriangle(vertex,tri1,tri2,tri3)
    pPrimeInfo = []
    if inside:
        distance = norm(vertex - tempPPrime)
        pPrime = tempPPrime
        distanceType = 1 
               
    else:
        dist1,tempPPrime1 = pointToLineSegmentDistance(vertex,tri1,tri2)
        dist2,tempPPrime2 = pointToLineSegmentDistance(vertex,tri2,tri3)
        dist3,tempPPrime3 = pointToLineSegmentDistance(vertex,tri3,tri1)
        dist4 = norm(vertex - tri1)
        dist5 = norm(vertex - tri2)
        dist6 = norm(vertex - tri3)
        tempDsts = [dist1,dist2,dist3,dist4,dist5,dist6]
        tempPPrimes = [tempPPrime1,tempPPrime2,tempPPrime3,tri1,tri2,tri3]
        
        indexes = [i for i, x in enumerate(tempDsts) if x == min(tempDsts)]
        if len(indexes) > 1:
            sys.exit("Warning: The algorithm to calculate the nearest distance from vertex to a triangle is wrong.") 
        
        distance = tempDsts[indexes[0]]
        pPrime = tempPPrimes[indexes[0]]

        if indexes[0] == 0:
            distanceType = 2
            pPrimeInfo.append(0)
            pPrimeInfo.append(1)
        elif indexes[0] == 1:
            distanceType = 2
            pPrimeInfo.append(1)
            pPrimeInfo.append(2)
        elif indexes[0] == 2:
            distanceType = 2
            pPrimeInfo.append(2)
            pPrimeInfo.append(0)
        elif indexes[0] == 3:
            distanceType = 3
            pPrimeInfo.append(0)
        elif indexes[0] == 4:
            distanceType = 3
            pPrimeInfo.append(1)
        else:
            distanceType = 3
            pPrimeInfo.append(2)
         
    # distanceType = 1: the vertex is located inside of an triangle of the mesh
    # distanceType = 2: the vertex is located on an edge of the mesh
    # distanceType = 3: the vertex is located on a vertex of the mesh
    return distance,pPrime,pPrimeInfo,distanceType

def rotateVector_Rodrigues(currentVector,rotationAxis,theta):
    # currentVector: current vector (3D vector) that is going to be rotated
    # rotationAxis: the axis (a 3D vector) that the currentVector is going to be rotated around
    # theta: the angle (a scaler) that the currentVector is going to be rotated around
    
    rotationAxis = np.array(rotationAxis)
    currentVector = np.array(currentVector)

    k = rotationAxis/norm(rotationAxis)
    # using Rodrigues' rotation formula to determine the velocity direction
    new3DVector = currentVector*np.cos(theta)+np.cross(k,currentVector)*np.sin(theta)+k*np.dot(k,np.transpose(currentVector))*(1-np.cos(theta))
    return new3DVector

def allMinsAndIndexesInList(a):
    smallest = min(a)
    return smallest, [index for index, element in enumerate(a) if smallest == element]

def checkParticleInside2dObject(particle,closestVertex,edge1Norm,edge2Norm,convexity):
    inside = 0
    vpVector3d = particle - closestVertex
    if convexity == 1:
        if dot(vpVector3d,edge1Norm) < 0 and dot(vpVector3d,edge2Norm) < 0:
            inside = 1 # the particle is inside the mesh
        elif dot(vpVector3d,edge1Norm) == 0 or dot(vpVector3d,edge2Norm) == 0:
            inside = 0 # the particle is on bc edges of the mesh
        else:
            inside = -1 # the particle is outside the mesh
    else:
        if dot(vpVector3d,edge1Norm) < 0 or dot(vpVector3d,edge2Norm) < 0:
            inside = 1 # the particle is inside the mesh
        elif dot(vpVector3d,edge1Norm) == 0 or dot(vpVector3d,edge2Norm) == 0:
            inside = 0 # the particle is on bc edges of the mesh
        else:
            inside = -1 # the particle is outside the mesh
    return inside


def generateTrussForBlender(allEdges,vp,offsetDistance):
    fileOutPut = open("TrussStructure.txt","w") 
    fileOutPut.write('radius = '+str(offsetDistance)+'\n') # first line is the radius
    for edge in allEdges:
        linkStartPoint = vp[edge[0]]
        linkEndPoint = vp[edge[1]]
        strStart = str(linkStartPoint[0]) + ',' + str(linkStartPoint[1]) + ','+ str(linkStartPoint[2])
        strEnd = strStart + ',' + str(linkEndPoint[0]) + ',' + str(linkEndPoint[1]) + ','+ str(linkEndPoint[2]) + '\n'
        fileOutPut.write(strEnd)

        #fileOutPut.write(str(linkStartPoint[0]),linkStartPoint[1], linkStartPoint[2],linkEndPoint[0] , linkEndPoint[1] , linkEndPoint[2]+' \n')
        #fileOutPut.write(strStart[1:-1]+' '+strEnd[1:-1]+' \n') # the following lines are the link start and end points
        #ax.plot([linkStartPoint[0],linkEndPoint[0]], [linkStartPoint[1],linkEndPoint[1]], [linkStartPoint[2],linkEndPoint[2]])
        #print(linkStartPoint)
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    #plt.show()
    fileOutPut.close()
    print("Generate Truss For Blender")


def generatePorousForBlender(allEdges,vp,radius):
    fileOutPut = open("TrussStructure.txt","w") 
    fileOutPut.write('radius = '+str(radius)+'\n') # first line is the radius
    for edge in allEdges:
        linkStartPoint = vp[edge[0]]
        linkEndPoint = vp[edge[1]]
        strStart = str(linkStartPoint[0]) + ',' + str(linkStartPoint[1]) + ','+ str(linkStartPoint[2])
        strEnd = strStart + ',' + str(linkEndPoint[0]) + ',' + str(linkEndPoint[1]) + ','+ str(linkEndPoint[2]) + '\n'
        fileOutPut.write(strEnd)

        #fileOutPut.write(str(linkStartPoint[0]),linkStartPoint[1], linkStartPoint[2],linkEndPoint[0] , linkEndPoint[1] , linkEndPoint[2]+' \n')
        #fileOutPut.write(strStart[1:-1]+' '+strEnd[1:-1]+' \n') # the following lines are the link start and end points
        #ax.plot([linkStartPoint[0],linkEndPoint[0]], [linkStartPoint[1],linkEndPoint[1]], [linkStartPoint[2],linkEndPoint[2]])
        #print(linkStartPoint)
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    #plt.show()
    fileOutPut.close()
    print("Generate Porous Structure For Blender")

def generateCloundPointForMatlab2D(particles):
    fileOutPut = open("loadData2D.m","w") 

    fileOutPut.write('function [pts] = readData() \n') # first line is the radius
    lenPP = len(particles)
    for idx,pp in enumerate(particles):
        if idx == 0:
            fileOutPut.write(' pts = ['+ str(pp[0]) +',' + str(pp[1]) + ';\n')
        elif idx == lenPP - 1:
            fileOutPut.write(str(pp[0]) +',' + str(pp[1]) + '];\n')
        else:
            fileOutPut.write(str(pp[0]) +',' + str(pp[1]) + ';\n')
    fileOutPut.write('end')    
        #ax.plot([linkStartPoint[0],linkEndPoint[0]], [linkStartPoint[1],linkEndPoint[1]], [linkStartPoint[2],linkEndPoint[2]])
        #print(linkStartPoint)
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    #plt.show()
    fileOutPut.close()
    print("Generate PPs For Matlab")

def getTriangleArea(p1,p2,p3):
    a = norm(p1-p2)
    b = norm(p2-p3)
    c = norm(p1-p3)

    s = (a + b + c)/2
    area = (s*(s-a)*(s-b)*(s-c)) ** 0.5    
    return area

def generateCloundPointForMatlab3D(particles):
    fileOutPut = open("loadData2D.m","w") 

    fileOutPut.write('function [pts] = loadData2D() \n') # first line is the radius
    lenPP = len(particles)
    for idx,pp in enumerate(particles):
        if idx == 0:
            fileOutPut.write(' pts = ['+ str(pp[0]) +',' + str(pp[1]) +',' + str(0.0) + ';\n')
        elif idx == lenPP - 1:
            fileOutPut.write(str(pp[0]) +',' + str(pp[1]) +',' + str(0.0) + '];\n')
        else:
            fileOutPut.write(str(pp[0]) +',' + str(pp[1]) +',' + str(0.0) + ';\n')
    fileOutPut.write('end')    
        #ax.plot([linkStartPoint[0],linkEndPoint[0]], [linkStartPoint[1],linkEndPoint[1]], [linkStartPoint[2],linkEndPoint[2]])
        #print(linkStartPoint)
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    #plt.show()
    fileOutPut.close()
    print("Generate 3D data For Matlab")

def shapeQuality(mesh,pps):
    qList = []
    for tri in mesh:
        p1 = pps[tri[0]]
        p2 = pps[tri[1]]
        p3 = pps[tri[2]]

        l1 = norm(p2-p1) 
        l2 = norm(p3-p1)
        l3 = norm(p3-p2)
        area = getTriangleArea(p1,p2,p3)

        q = 4*np.sqrt(3)*area/(l1**2+l2**2+l3**2)
        qList.append(q)

    return qList

def exportTriangleQualityAndMesh2D(mesh,pps,qList):
    fileOutPut = open("loadualityAndMesh2D.m","w") 

    fileOutPut.write('function [mesh,pts,qualities] = loadualityAndMesh2D() \n') # first line is the radius
    lenPP = len(pps)
    for idx,pp in enumerate(pps):
        if idx == 0:
            fileOutPut.write('pts = ['+ str(pp[0]) +',' + str(pp[1]) + ';\n')
        elif idx == lenPP - 1:
            fileOutPut.write(str(pp[0]) +',' + str(pp[1]) + '];\n')
        else:
            fileOutPut.write(str(pp[0]) +',' + str(pp[1]) + ';\n')
    fileOutPut.write('\n')
    
    lenMesh = len(mesh)
    for idx,tri in enumerate(mesh):
        if idx == 0:
            fileOutPut.write('mesh = ['+ str(tri[0]+1) +',' + str(tri[1]+1) +',' + str(tri[2]+1) + ';\n')
        elif idx == lenMesh - 1:
            fileOutPut.write(str(tri[0]+1) +',' + str(tri[1]+1) +',' + str(tri[2]+1) + '];\n')
        else:
            fileOutPut.write(str(tri[0]+1) +',' + str(tri[1]+1) +',' + str(tri[2]+1) + ';\n')
    fileOutPut.write('\n')

    lenqList = len(qList)
    for idx,q in enumerate(qList):
        if idx == 0:
            fileOutPut.write('qualities = ['+ str(q) + ';\n')
        elif idx == lenqList - 1:
            fileOutPut.write(str(q) + '];\n')
        else:
            fileOutPut.write(str(q) + ';\n')
    fileOutPut.write('end')
        #ax.plot([linkStartPoint[0],linkEndPoint[0]], [linkStartPoint[1],linkEndPoint[1]], [linkStartPoint[2],linkEndPoint[2]])
        #print(linkStartPoint)
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    #plt.show()
    fileOutPut.close()
    print("Generate (mesh,pts,qualities) For Matlab")

def exportConvergenceForCoinMesh2D(numberOfPpps,distRatioList):
    fileOutPut = open("loadConvergenceInoForCoin2dMesh.m","w") 

    fileOutPut.write('function [numberOfPpps,distRatioList] = loadConvergenceInoForCoin2dMesh() \n') # first line is the radius
    lenPps = len(numberOfPpps)
    for idx,numOfpp in enumerate(numberOfPpps):
        if idx == 0:
            fileOutPut.write('numberOfPpps = ['+ str(numOfpp) + ';\n')
        elif idx == lenPps - 1:
            fileOutPut.write(str(numOfpp) + '];\n')
        else:
            fileOutPut.write(str(numOfpp) + ';\n')
    fileOutPut.write('\n')
    
    lenDistRatio = len(distRatioList)
    for idx,distRatio in enumerate(distRatioList):
        if idx == 0:
            fileOutPut.write('distRatioList = ['+ str(distRatio) + ';\n')
        elif idx == lenDistRatio - 1:
            fileOutPut.write(str(distRatio) + '];\n')
        else:
            fileOutPut.write(str(distRatio) + ';\n')
    fileOutPut.write('\n')
   
    fileOutPut.write('end')
        #ax.plot([linkStartPoint[0],linkEndPoint[0]], [linkStartPoint[1],linkEndPoint[1]], [linkStartPoint[2],linkEndPoint[2]])
        #print(linkStartPoint)
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    #plt.show()
    fileOutPut.close()
    print("Generate (ConvergenceForCoinMesh2D) For Matlab")


def exportConvergenceAndEdgeLengthErrors(numberOfPpps,distRatioList,edgeLengthErrorList):
    fileOutPut = open("loadConvergenceInoAndEdgeLengthErrorsFor2dMesh.m","w") 

    #fileOutPut.write('function [numberOfPpps,distRatioList,edgeLengthErrorList] = loadConvergenceInoAndEdgeLengthErrorsFor2dMesh() \n') # first line is the radius
    lenPps = len(numberOfPpps)
    for idx,numOfpp in enumerate(numberOfPpps):
        if idx == 0:
            fileOutPut.write('numberOfPpps = ['+ str(numOfpp) + ';\n')
        elif idx == lenPps - 1:
            fileOutPut.write(str(numOfpp) + '];\n')
        else:
            fileOutPut.write(str(numOfpp) + ';\n')
    fileOutPut.write('\n')
    
    lenDistRatio = len(distRatioList)
    for idx,distRatio in enumerate(distRatioList):
        if idx == 0:
            fileOutPut.write('distRatioList = ['+ str(distRatio) + ';\n')
        elif idx == lenDistRatio - 1:
            fileOutPut.write(str(distRatio) + '];\n')
        else:
            fileOutPut.write(str(distRatio) + ';\n')
    fileOutPut.write('\n')
    
    lenDistRatio = len(edgeLengthErrorList)
    for idx,distRatio in enumerate(edgeLengthErrorList):
        if idx == 0:
            fileOutPut.write('edgeLengthErrorList = ['+ str(distRatio) + ';\n')
        elif idx == lenDistRatio - 1:
            fileOutPut.write(str(distRatio) + '];\n')
        else:
            fileOutPut.write(str(distRatio) + ';\n')
    fileOutPut.write('\n')

    fileOutPut.write('end')
        #ax.plot([linkStartPoint[0],linkEndPoint[0]], [linkStartPoint[1],linkEndPoint[1]], [linkStartPoint[2],linkEndPoint[2]])
        #print(linkStartPoint)
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    #plt.show()
    fileOutPut.close()
    print("Generate (ConvergenceInoAndEdgeLengthErrorsFor2dMesh) For Matlab")

def export2dTrussForAbaqus(pts,edges,InputFileName):
    filename = "2dTruss"+InputFileName+".inp"
    fileOutPut = open(filename,"w") 

    pts = pts*0.01
    #fileOutPut.write('function [numberOfPpps,distRatioList] = loadConvergenceInoForCoin2dMesh() \n') # first line is the radius
    fileOutPut.write('*Part, name=') # first line is the radius
    modelName = "Truss2d"+InputFileName
    fileOutPut.write(modelName + '\n')
    fileOutPut.write('*Node \n')
    #lenPps = len(pts)
    for idx,pt in enumerate(pts):
        fileOutPut.write('      '+ str(idx+1) + ',   ' +  str(np.round(pt[0],5)) + ',   ' +  str(np.round(pt[1],5)) + '\n')
        #if idx == 0:
        #    fileOutPut.write('numberOfPpps = ['+ str(numOfpp) + ';\n')
        #elif idx == lenPps - 1:
        #    fileOutPut.write(str(numOfpp) + '];\n')
        #else:
        #    fileOutPut.write(str(numOfpp) + ';\n')
    #fileOutPut.write('\n')
    fileOutPut.write('*Element, type=T2D2 \n') 
    #lenDistRatio = len(edges)
    for idx,edge in enumerate(edges):
        fileOutPut.write(str(idx+1) + ', ' +  str(edge[0]+1) + ', ' +  str(edge[1]+1) + '\n')
        #if idx == 0:
        #    fileOutPut.write('distRatioList = ['+ str(distRatio) + ';\n')
        #elif idx == lenDistRatio - 1:
        #    fileOutPut.write(str(distRatio) + '];\n')
        #else:
        #    fileOutPut.write(str(distRatio) + ';\n')
    fileOutPut.write('\n')
   
    fileOutPut.write('*End Part')
        #ax.plot([linkStartPoint[0],linkEndPoint[0]], [linkStartPoint[1],linkEndPoint[1]], [linkStartPoint[2],linkEndPoint[2]])
        #print(linkStartPoint)
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    #plt.show()
    fileOutPut.close()
    print("Generate (export2dTrussForAbaqus) For Abaqus")
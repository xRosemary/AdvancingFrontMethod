# the OBJ model must be composed of only triangle faces and is exported by the free software Meshmixer.
# The best way to get an OBJ file that is compatable to the FLOW MESH is convert a STL file to an OBJ file using Meshmixer.

import numpy as np

from MathTools import rotateVector_Rodrigues
from MathTools import norm

def read2dObjShap(Name):
    # Read an obj file. The obj file should be placed in the code's folder.
    # v: Vertices positions    
    # vn: the normal vectors of the vertices
    # mesh: the triangles of the mesh. It includes the vertex IDs of the mesh
    # meshNorm: the normal vectors of triangles of the mesh    
    import os
    from pathlib import Path    

    direction = os.path.dirname(__file__)
    fileName = os.path.join(direction,Name)    
    my_file = Path(fileName)

    if my_file.is_file():
        with open(fileName) as f:
            content = f.readlines()
            f.close()
                
        v1=[]
        v2=[]
        v3=[]
        vn1=[]
        vn2=[]
        vn3=[]
        mesh = []
        for line in content:
            linesplit = line.split()
            if linesplit[0] == 'v':
                v1.append(float(linesplit[1]))
                v2.append(float(linesplit[2]))
                v3.append(float(linesplit[3]))
            elif linesplit[0] == 'vn':
                vn1.append(float(linesplit[1]))
                vn2.append(float(linesplit[2]))
                vn3.append(float(linesplit[3]))
            elif linesplit[0] == 'f':        
                meshSplit1 = linesplit[1].split("//")
                meshSplit2 = linesplit[2].split("//")
                meshSplit3 = linesplit[3].split("//")
                mesh.append([int(meshSplit1[0]),int(meshSplit2[0]),int(meshSplit3[0])])
        v = [v1,v2,v3]
        v = np.array(v)
        v = np.transpose(v)
        vn = [vn1,vn2,vn3]
        vn = np.array(vn)
        vn = np.transpose(vn)
                
        mesh = np.array(mesh)
        mesh = mesh - 1 # the index value started in obj file is 1, while it is 0 in python array data
        
        print('Successfully read the file.')
        objectNorm = vn[1,:]
        return v,objectNorm,mesh
    else:
        print('File does not exist inside the foler.')
        return [],[],[]

def obtainBcEdges(mesh):
    # BC edges are only shared by one triangles
    edgesIDsDict = {}
    edgesSharedNum = {}
    for tri in mesh:        
        edge1 = str(tri[0])+"|"+str(tri[1])
        edge2 = str(tri[1])+"|"+str(tri[2])
        edge3 = str(tri[2])+"|"+str(tri[0]) 
        if edge1 in edgesSharedNum:
            edgesSharedNum[edge1] = edgesSharedNum[edge1] + 1
        else:
            edgesSharedNum[edge1] = 1
        
        if edge2 in edgesSharedNum:
            edgesSharedNum[edge2] = edgesSharedNum[edge2] + 1
        else:
            edgesSharedNum[edge2] = 1
        
        if edge3 in edgesSharedNum:
            edgesSharedNum[edge3] = edgesSharedNum[edge3] + 1
        else:
            edgesSharedNum[edge3] = 1        
        edgesIDsDict.setdefault(edge1,[tri[0],tri[1]])
        edgesIDsDict.setdefault(edge2,[tri[1],tri[2]])
        edgesIDsDict.setdefault(edge3,[tri[2],tri[0]])

    bcEdges = []
    bcEdgesDict = {}
    for edge in edgesSharedNum:
        edgeSharedNum = edgesSharedNum[edge]
        edgeVerticesID = edge.split("|")
        edge1 = edgeVerticesID[1]+"|"+edgeVerticesID[0]        
        if edge1 in edgesSharedNum:
            edgeSharedNum = edgeSharedNum + edgesSharedNum[edge1]        
        if edgeSharedNum == 1:            
            bcEdges.append(edgesIDsDict[edge])
            bcEdgesDict.setdefault(edge,edgesIDsDict[edge])
    # bcEdges include vertices' IDs
    return bcEdges
    

def obtainBcVertices(mesh):
    bcEdges = obtainBcEdges(mesh)
    bcVerticesIDs = []
    for edge in bcEdges:
        bcVerticesIDs.extend(edge)
    bcVerticesIDs = list(dict.fromkeys(bcVerticesIDs))    

    return bcVerticesIDs


def findBcEdgesNorms(Vertices,bcVerticesIDs,bcEdges,objectNorm):
    pi = 3.1415926

    rotationAxis = objectNorm
    theta = -pi/2 # clockwise direction
    
    bcEdgesNorms = []
    for bcEdge in bcEdges:
        vector3d = Vertices[bcEdge[1]] - Vertices[bcEdge[0]]
        vector3d = vector3d/norm(vector3d)
        newVector3d = rotateVector_Rodrigues(vector3d,rotationAxis,theta)
        bcEdgesNorms.append(newVector3d)

    return bcEdgesNorms

def obtainBcVerticesIDsToBcEdgesIDsDict(bcEdges,bcVerticesIDs):
    # obtain the dictionary:  Dict[key: vertex ID] = the IDs of the two edges' sharing the vertex
    dictBcVerticesIDstoEdgesIDs = {}
    for idx,bcEdge in enumerate(bcEdges):
        vertexID = bcEdge[0]        
        if vertexID in dictBcVerticesIDstoEdgesIDs:
            edgeID = dictBcVerticesIDstoEdgesIDs[vertexID]
            dictBcVerticesIDstoEdgesIDs[vertexID] = [edgeID,idx]
        else:
            dictBcVerticesIDstoEdgesIDs[vertexID] = idx

        vertexID = bcEdge[1]
        if vertexID in dictBcVerticesIDstoEdgesIDs:
            edgeID = dictBcVerticesIDstoEdgesIDs[vertexID]
            dictBcVerticesIDstoEdgesIDs[vertexID] = [edgeID,idx]
        else:
            dictBcVerticesIDstoEdgesIDs[vertexID] = idx
    return dictBcVerticesIDstoEdgesIDs

def generateClouldPointsFile(pps):
    fileOutPut = open("ParticlesPositions.txt","w") 
    for pp in pps:
        decimalnum = 6
        pp0 = round(pp[0], decimalnum)
        pp1 = round(pp[1], decimalnum)
        pp2 = round(pp[2], decimalnum)
        fileOutPut.write(str(pp0)+' '+str(pp1)+' '+str(pp2)+'\n')
    fileOutPut.close()
    print("Generate clould point file")



def findAllEdges2dTriangularMesh(mesh):
    allEdgesDict = {}
    for tri in mesh:
        tri.sort()
        edge1 = str(tri[0])+"|"+str(tri[1])
        edge2 = str(tri[0])+"|"+str(tri[2])
        edge3 = str(tri[1])+"|"+str(tri[2])             
        allEdgesDict.setdefault(edge1,[tri[0],tri[1]])
        allEdgesDict.setdefault(edge2,[tri[0],tri[2]])
        allEdgesDict.setdefault(edge3,[tri[1],tri[2]])
    allEdges = list(allEdgesDict.values())
    return allEdges
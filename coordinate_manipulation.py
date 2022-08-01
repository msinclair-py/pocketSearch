#!/usr/bin/env python
import numpy as np
from typing import List
from scipy.spatial.transform import Rotation as R


class Coordinates:
    """
    This class is intended to perform a variety of point cloud manipulations.
    """
    
    def __init__(self, directory: str, pdb: str, pnum: int = 0, deg1: int = 0, 
                    deg2: int = 0, translation: int = 0, rotate: float = 0., 
                    mode: int = 1) -> None:
        
        self.directory = directory
        self.pdb = pdb
        self.pnum = pnum
        self.deg1 = deg1
        self.deg2 = deg2
        self.tr = translation
        self.rotates = rotate
        self.mode = mode
    
    
    #extract coordinates from pdb file, given the limited size of each pocket pdb
    def get_coords(self) -> None:
        if not self.pnum:
            fpath = f'{self.directory}{self.pdb}.pocket.pdb'
        else:
            fpath = f'{self.directory}{self.pdb}.{self.pnum}.pdb'
    
        with open(fpath) as f:
            lines = f.readlines()
            initcoords = np.zeros((len(lines),3))

            for i in range(initcoords.shape[0]):
                l = lines[i]
                initcoords[i] = [l[30:38].strip(),l[38:46].strip(),l[46:54].strip()]

        self.coords = initcoords
    
    
    #obtain the longest principal axis through coordinate set
    def principal(self) -> None:
        inertia = self.coords.T @ self.coords
        e_values, e_vectors = np.linalg.eig(inertia)
        order = np.argsort(e_values)
        eval3,eval2,eval1 = e_values[order]
        axis3,axis2,axis1 = e_vectors[:,order].T
        self.principal = axis1
    
    
    #center array based on geometric center of mass
    def center(self) -> None:
        array = self.coords
        centered = np.zeros((array.shape[0],array.shape[1]))
        com = np.mean(array, axis=0)
        
        for i in range(array.shape[0]):
            centered[i] = array[i] - com

        self.coords = centered
    
    
    #obtain rotation matrix using proof, then apply to coordinate system
    #"a" corresponds to a unit vector along the longest principal axis
    def align(self) -> None:
        array = self.coords
        a = self.principal
        aligned=np.zeros((array.shape[0],array.shape[1]))
        b = [0,0,1]
        v = np.cross(a, b)
        c = np.dot(a, b)
        I = np.eye(3, 3)
        vx = np.array([[0,   -v[2],  v[1]],
                        [v[2],   0,  -v[0]],
                        [-v[1], v[0],   0]])
        rotmatrix = I + vx + (vx @ vx)/(1+c)
        aligned = rotmatrix @ array.T
        self.coords = aligned.T
    
    
    #create pdb, keeping relevant conformational state data
    def make_pdb(self, array, a = None, b = None, c = None, d = None, e = None) -> None:
        if self.mode == 1:
            with open(f'{self.directory}aligned.{self.pdb}.{self.pnum}','w') as f:
                with open(f'{self.directory}{self.pdb}.{self.pnum}.pdb','r') as infile:
                    i=0
                    for line in infile:
                        x = f'{array[i][0]:.3f}'
                        y = f'{array[i][1]:.3f}'
                        z = f'{array[i][2]:.3f}'
                        f.write(f'{line[:27]}{x:>11}{y:>8}{z:>8} {line[55:]}')
                        i+=1
    
        elif self.mode == 0:                
            with open(f'{self.directory}{self.pdb}_{a}_{b}_{c}_{d}_{e}','w') as f:
                with open(f'{self.directory}{self.pdb}.pocket.pdb','r') as infile:
                    i=0
                    for line in infile:
                        x=f'{array[i][0]:.3f}'
                        y=f'{array[i][1]:.3f}'
                        z=f'{array[i][2]:.3f}'
                        f.write(f'{line[:27]}{x:>11}{y:>8}{z:>8} {line[55:]}')
                        i+=1

        f.close()
        infile.close()
    
    
    #conformation generation/flow control function
    def conformation(self, conformations):
        array = self.coords
        for i in range(len(conformations)):
            deg = conformations[i]
            conform = R.from_euler('z', deg, degrees=True)
            conformation = conform.apply(array)
            self.make_pdb(conformation,f'conf{i}',0,0,0,0)
            self.tilt(conformation,f'conf{i}',0,0,0)
            self.flip(conformation,f'conf{i}',0)
            self.translate(conformation,f'conf{i}')
    
    
    #rotation function
    def rotate(self,array,conf,tilted,trans,flipped):
        rotated_array=np.array([])
    
        for i in range(1,self.rotates+1):
            deg = i*int(self.deg1)
            z_rot = R.from_euler('z',deg,degrees=True)
            rotated_array = z_rot.apply(array)
            self.make_pdb(rotated_array,conf,deg,tilted,trans,flipped)
    
        if tilted == 1:
            self.tilt(array,conf,tilted,trans,flipped)
    
    
    #tilt function
    def tilt(self,array,conf,tlt,trans,flipped):
        tlt+=1
        tilted_array=np.array([])
        tilter = R.from_euler('x', self.deg2, degrees=True)
        tilted_array = tilter.apply(array)
        self.rotate(tilted_array,conf,tlt,trans,flipped)
    
    
    #translation function
    def translate(self,arr,conf):
        xpos=np.zeros((arr.shape[0],arr.shape[1]))
        xneg=np.zeros((arr.shape[0],arr.shape[1]))
        ypos=np.zeros((arr.shape[0],arr.shape[1]))
        yneg=np.zeros((arr.shape[0],arr.shape[1]))
        zpos=np.zeros((arr.shape[0],arr.shape[1]))
        zneg=np.zeros((arr.shape[0],arr.shape[1]))
        x=np.array([self.tr,0,0])
        y=np.array([0,self.tr,0])
        z=np.array([0,0,self.tr])
    
        for i in range(len(arr)):
            xpos[i]=arr[i]+x
            xneg[i]=arr[i]-x
            ypos[i]=arr[i]+y
            yneg[i]=arr[i]-y
            zpos[i]=arr[i]+z
            zneg[i]=arr[i]-z
    
        #iterate through list of translations, performing all previous
        #conformational changes
        translations = [xpos,xneg,ypos,yneg,zpos,zneg]
        for i in range(len(translations)):
            self.make_pdb(translations[i],conf,0,0,i+1,0)
            self.tilt(translations[i],conf,0,i+1,0)
            self.flip(translations[i],conf,i+1)
    
    
    #180 degree flip function
    def flip(self,arr,conf,trans):
        flp=np.array([])
        f = R.from_euler('x', 180, degrees=True)
        flp = f.apply(arr)
        self.make_pdb(flp,conf,0,0,trans,1)
        self.tilt(flp,conf,0,trans,1)

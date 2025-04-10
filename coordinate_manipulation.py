#!/usr/bin/env python
from copy import deepcopy
import numpy as np
from typing import List, Union
from scipy.spatial.transform import Rotation as R

PathLike = Union[str, Path]
OptStr = Union[str, None]

class Coordinates:
    """
    This class is intended to perform a variety of point cloud manipulations.
    """
    def __init__(self, 
                 directory: PathLike, 
                 pdb: PathLike, 
                 pnum: int = 0, 
                 deg1: int = 0, 
                 deg2: int = 0, 
                 translation: int = 0, 
                 rotate: float = 0., 
                 mode: int = 1) -> None:
        self.directory = directory
        self.pdb = pdb
        self.pnum = pnum
        self.deg1 = deg1
        self.deg2 = deg2
        self.tr = translation
        self.rotates = rotate
        self.mode = mode
    
    def get_coords(self) -> None:
        """
        Extracts coordinates from pdb file.
        """
        basepath = self.directory / self.pdb
        ext = f'.{self.pnum}.pdb' if self.pnum else '.pocket.pdb'
        fpath = basepath.with_suffix(ext)
    
        with open(fpath) as f:
            lines = f.readlines()
            initcoords = np.zeros((len(lines), 3))

            for i in range(initcoords.shape[0]):
                l = lines[i]
                initcoords[i] = [l[30:38].strip(), l[38:46].strip(), l[46:54].strip()]

        self.coords = initcoords
    
    def principal(self) -> None:
        """
        Obtain the longest principal axis through coordinate set
        """
        inertia = self.coords.T @ self.coords
        e_values, e_vectors = np.linalg.eig(inertia)
        order = np.argsort(e_values)
        eval3, eval2, eval1 = e_values[order]
        axis3, axis2, axis1 = e_vectors[:, order].T
        self.principal = axis1
    
    def center(self) -> None:
        """
        Center array based on geometric center of mass
        """
        self.coords -= np.mean(self.coords, axis=0)
    
    def align(self) -> List[List[float]]:
        """
        Obtain rotation matrix using proof, then apply to coordinate system.
        "a" corresponds to a unit vector along the longest principal axis
        """
        array = self.coords
        a = self.principal
        b = np.array([0, 0, 1])
        v = np.cross(a, b)
        c = np.dot(a, b)
        I = np.eye(3, 3)
        
        vx = np.array(
            [
                [0,    -v[2],   v[1]],
                [v[2],     0,  -v[0]],
                [-v[1], v[0],      0]
            ]
        )

        rotmatrix = I + vx + (vx @ vx) / (1 + c)
        aligned = rotmatrix @ self.coords.T
        self.coords = aligned.T
        return self.coords
    
    def make_pdb(self, 
                 array: np.ndarray, 
                 a: OptStr=None, 
                 b: OptStr=None, 
                 c: OptStr=None, 
                 d: OptStr=None,
                 e: OptStr=None) -> None:
        """
        Create pdb, keeping relevant conformational state data
        """
        if self.mode == 1:
            fout = open(self.directory / f'aligned.{self.pdb}.{self.pnum}', 'w')
            fin = open(self.directory / f'{self.pdb}.{self.pnum}.pdb', 'r')
    
        elif self.mode == 0:                
            fout = open(self.directory / f'{self.pdb}_{a}_{b}_{c}_{d}_{e}','w')
            fin = open(self.directory f'{self.pdb}.pocket.pdb','r')

        for i, line in enumerate(fin):
            x=f'{array[i][0]:.3f}'
            y=f'{array[i][1]:.3f}'
            z=f'{array[i][2]:.3f}'

            fout.write(f'{line[:27]}{x:>11}{y:>8}{z:>8} {line[55:]}')

        fout.close()
        fin.close()
    
    def conformation(self, 
                     conformations: np.ndarray) -> None:
        """
        Conformation generation/flow control function
        """
        array = deepcopy(self.coords)
        for i in range(len(conformations)):
            deg = conformations[i]
            conform = R.from_euler('z', deg, degrees=True)
            conformation = conform.apply(array)
            self.make_pdb(conformation, f'conf{i}', 0, 0, 0, 0)
            self.tilt(conformation, f'conf{i}', 0, 0, 0)
            self.flip(conformation, f'conf{i}', 0)
            self.translate(conformation, f'conf{i}')
    
    
    def rotate(self,
               array: np.ndarray,
               conf: str,
               tilted: int,
               trans: int,
               flipped: int) -> None:
        """
        Performs rotations
        """
        rotated_array=np.array([])
    
        for i in range(1, self.rotates + 1):
            deg = i * int(self.deg1)
            z_rot = R.from_euler('z', deg, degrees=True)
            rotated_array = z_rot.apply(array)
            self.make_pdb(rotated_array, conf, deg, tilted, trans, flipped)
    
        if tilted == 1:
            self.tilt(array, conf, tilted, trans, flipped)
    
    
    def tilt(self,
             array: np.ndarray, 
             conf: str,
             tlt: int,
             trans: int,
             flipped: int) -> None:
        """
        Tilts along the x-axis by `self.deg2` degrees
        """
        tlt+=1
        tilted_array=np.array([])
        tilter = R.from_euler('x', self.deg2, degrees=True)
        tilted_array = tilter.apply(array)
        self.rotate(tilted_array, conf, tlt, trans, flipped)
    
    def translate(self,
                  arr: np.ndarray,
                  conf: str) -> None:
        """
        Translates coordinates
        """
        x = np.array([self.tr,0,0])
        y = np.array([0,self.tr,0])
        z = np.array([0,0,self.tr])
    
        translations = [
            arr + x,
            arr - x,
            arr + y,
            arr - y,
            arr + z,
            arr - z
        ]
        
        #iterate through list of translations, performing all previous
        #conformational changes
        for i, translation in enumerate(translations):
            self.make_pdb(translation, conf, 0, 0, i+1, 0)
            self.tilt(translation, conf, 0, i+1, 0)
            self.flip(translation, conf, i+1)
    
    def flip(self,
             arr: np.ndarray,
             conf: str,
             trans: int) -> None:
        """
        Flips coordinate system by 180 degrees about the x-axis
        """
        flp = np.array([])
        f = R.from_euler('x', 180, degrees=True)
        flp = f.apply(arr)
        self.make_pdb(flp, conf, 0, 0, trans, 1)
        self.tilt(flp, conf, 0, trans, 1)

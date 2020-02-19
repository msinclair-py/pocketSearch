#!/usr/bin/env python
import numpy as np
from scipy.spatial.transform import Rotation as R


class Coordinates:

	def __init__(self,directory,pdb,deg1=None,deg2=None,translation=None,rotate=None,mode=1):
		
		self.directory = directory
		self.pdb = pdb
		self.deg1 = deg1
		self.deg2 = deg2
		self.tr = translation
		self.rotates = rotate
		self.mode = mode


	#extract coordinates from pdb file, given the limited size of each pocket pdb
	#there should be no issues accessing them by column index
	def getCoords(self):
		initcoords = np.genfromtxt(f'{self.directory}{self.pdb}.pocket.pdb',
			dtype=float,usecols=(6,7,8))
		return initcoords

	
	#obtain the longest principal axis through coordinate set
	def principal(self,array):
		inertia = array.T @ array
		e_values, e_vectors = np.linalg.eig(inertia)
		order = np.argsort(e_values)
		eval3,eval2,eval1 = e_values[order]
		axis3,axis2,axis1 = e_vectors[:,order].T
		return axis1


	#center array based on geometric center of mass
	def center(self,array):
		centered = np.zeros((array.shape[0],array.shape[1]))
		com = array.mean(0)
		for i in range(array.shape[0]):
			centered[i] = array[i] - com
		return centered


	#obtain rotation matrix using proof, then apply to coordinate system
	#"a" corresponds to a unit vector along the longest principal axis
	def align(self,array,a):
		aligned=np.zeros((array.shape[0],array.shape[1]))
		b = [0,0,1]
		v = np.cross(a,b)
		c = np.dot(a,b)
		I = np.eye(3,3)
		vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
		rotmatrix = I + vx + (vx @ vx)/(1+c)
		aligned = rotmatrix @ array.T
		return aligned.T


	#create pdb, keeping relevant conformational state data
	def makePDB (self,array,a=None,b=None,c=None,d=None,e=None):
		if self.mode == 1:
			with open(f'{self.directory}aligned.{self.pdb}','w') as f:
				with open(f'{self.directory}{self.pdb}','r') as infile:
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


	#conformation generation/flow control function
	def conformation(self,array,conformations):
		for i in range(len(conformations)):
			deg = conformations[i]
			conform = R.from_euler('z',deg,degrees=True)
			conformation = conform.apply(array)
			self.makePDB(conformation,f'conf{i}',0,0,0,0)
			self.tilt(conformation,f'conf{i}',0,0,0)
			self.flip(conformation,f'conf{i}',0)
			self.translate(conformation,f'conf{i}')


	#rotation function
	def rotate(self,array,conf,tilted,trans,flipped):
		rotated_array=np.array([])

		for i in range(self.rotates):
			deg = i*int(self.deg1)
			z_rot = R.from_euler('z',deg,degrees=True)
			rotated_array = z_rot.apply(array)
			self.makePDB(rotated_array,conf,deg,tilted,trans,flipped)

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
		for i in range(1,len(translations)):
			self.makePDB(translations[i],conf,0,0,i,0)
			self.tilt(translations[i],conf,0,i,0)
			self.flip(translations[i],conf,i)


	#180 degree flip function
	def flip(self,arr,conf,trans):
		flp=np.array([])
		f = R.from_euler('x', 180, degrees=True)
		flp = f.apply(arr)
		self.makePDB(flp,conf,0,0,trans,1)
		self.tilt(flp,conf,0,trans,1)

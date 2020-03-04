#!/usr/bin/env python
import os, glob, subprocess, random, Bio
import numpy as np
from aliases import fpocket,surf,vasp,pdblist
from Bio.PDB import PDBList

def randomPDBs(directory,rand):
	pdb1 = PDBList()
	PDB2list=[]
	array=[]
	with open(pdblist,'r') as infile:
		for line in infile:
			array.append(line)
	
	samp = random.sample(range(0,len(array)), rand)
	for num in samp:
		PDB2list.append(array[num].strip())
	for i in PDB2list:
		pdb1.retrieve_pdb_file(i,pdir=directory,file_format='pdb')


def checkFormat(directory):
	if directory[-1] != '/':
		directory = f'{directory}/'
	if not os.path.exists(directory):
		print('Directory doesn\'t exist')
	else:
		print('Directory exists')
	return directory


def rename(directory):
	for f in glob.glob(f'{directory}/*'):
		old = os.path.basename(f)
		if old[-4:] == '.ent':
			new = f'{old[3:-4]}.pdb'
			os.rename(f,f'{directory}{new}')


def principal (array):
	inertia = array.T @ array
	e_values, e_vectors = np.linalg.eig(inertia)
	order = np.argsort(e_values)
	eval3,eval2,eval1 = e_values[order]
	axis3,axis2,axis1 = e_vectors[:,order].T
	return axis1


def align (array,a):
	aligned=np.zeros((array.shape[0],array.shape[1])) 
	b = [0,0,1] 
	v = np.cross(a,b) 
	c = np.dot(a,b) 
	I = np.eye(3,3)
	vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
	R = I + vx + (vx @ vx)/(1+c)
	aligned = R @ array.T
	return aligned.T	


def center (array):
	centered=np.zeros((array.shape[0],array.shape[1]))
	com=array.mean(0)
	for i in range(array.shape[0]):
		centered[i] = array[i] - com
	return centered


def make_pdb (array,output):
	with open(f'{output}','w') as f:
		with open(f'{input}','r') as infile:
			i=0
			for line in infile:
				x = f'{array[i][0]:.3f}'
				y = f'{array[i][1]:.3f}'
				z = f'{array[i][2]:.3f}'
				f.write(f'{line[:27]}{x:>11}{y:>8}{z:>8} {line[55:]}')
				i+=1


def unique(list1):
	unique_list=[]
	for x in list1:
		if x not in unique_list:
			unique_list.append(x)
	return unique_list


def find_pockets(indir,alphas):
	for i in glob.glob(f'{indir}*.pdb'):
		args=(fpocket,'-f',i,'-i',alphas)
		str_args=[ str(x) for x in args ]
		subprocess.run(str_args)
	
	for i in glob.glob(f'{indir}*_out'):
		if len(os.listdir(i)) == 0:
			print(f'{i}: empty directory')
		else:
			print(f'{i}: has pockets')


def gen_surfs(outdir,indir):
	for pock in glob.glob(f'{indir}aligned.*'):
		pocket = os.path.basename(pock)
		n = pocket.split('.')[1]
		p = pocket.split('.')[2]
		arg = (surf,'-surfProbeGen',pock,f'{outdir}VASP/pockets/{n}_{p}.SURF',3,.5)
		str_arg = [ str(x) for x in arg ]
		subprocess.run(str_arg)


def intersect(outdir,initial):
	for struc in glob.glob(f'{outdir}VASP/pockets/*.SURF'):
		pocket = os.path.basename(struc)
		n = pocket.split('.')[0]
		for init in glob.glob(f'{initial}*.SURF'):
			fname = os.path.basename(init).split('.')
			conf = fname[1]
			rot = fname[2]
			tilt = fname[3]
			trans = fname[4]
			flip = fname[5]
			arg = (vasp,'-csg',struc,init,'I',f'{outdir}VASP/intersect.{n}.{conf}.{rot}.{tilt}.{trans}.{flip}.SURF',.5)
			str_arg = [ str(x) for x in arg ]
			subprocess.run(str_arg)

		arg2 = (surf,'-surveyVolume',struc)
		str_arg2 = [ str(x) for x in arg2 ]
		out = open(f'{outdir}VASP/pockets/{n}.vol.txt','w')
		subprocess.run(str_arg2, stdout=out)


def extract_score(outdir):
	for inter in glob.glob(f'{outdir}VASP/intersect*'):
		p = os.path.basename(inter).split('.')
		na = p[1]
		co = p[2]
		ro = p[3]
		ti = p[4]
		tr = p[5]
		fl = p[6]
		arg = (surf,'-surveyVolume',inter)
		out = open(f'{outdir}VASP/scores/{na}_{co}_{ro}_{ti}_{tr}_{fl}_iscore.txt','w')
		str_arg = [ str(x) for x in arg ]
		subprocess.run(str_arg, stdout=out)


def original_volume(outdir):
	for v in glob.glob(f'{outdir}VASP/pockets/*.SURF'):
		n = os.path.basename(v).split('.')[0]
		arg = (surf,'-surveyVolume',v)
		out = open(f'{outdir}VASP/pockets/{n}.vol.txt','w')
		str_arg = [ str(x) for x in arg ]
		subprocess.run(str_arg, stdout=out)


def generate_scorefile(outdir,initial,filt):
	files,pdbs,name_array,inter,pockets=[],[],[],[],[]
	for name in glob.glob(f'{outdir}VASP/scores/*iscore.txt'):
		score = os.path.basename(name)
		files.append(score)
		if score not in pdbs: #what is pdbs for???################################
			pdbs.append(score.split('_')[0])
	
	for name in files:
		name_array.append(name.split('_'))
	
	print('-----Getting Scores-----')
	for i, line in enumerate(name_array):
		name = f'{outdir}VASP/scores/{files[i]}'
		a=name_array[i]
		with open(name,'r') as infile:
			for line in infile:
				pass
			iline = float(line.split()[3])
			inter.append([a[0],a[1],a[2],a[3],a[4],a[5],a[6],iline])

	print('-----Sorting Scores-----')
	inter_sort,inter_final=[],[]
	inter_sort = sorted(inter,key=lambda x: x[-1], reverse=True)

	with open(f'{initial}vol.txt','r') as f:
		for line in f:
			pass
		vline = line.split()[-1]
	
	vols={}
	for p in inter_sort:
		with open(f'{outdir}VASP/pockets/{p[0]}_{p[1]}.vol.txt','r') as f:
			for line in f:
				pass
			key = f'{p[0]} {p[1]}'
			vol = line.split()[-1]
			vols.update({key: vol})
	
	tracker=[]
	hits={}

	for j in range(len(inter_sort)):
		key = f'{inter_sort[j][0]} {inter_sort[j][1]}'
		if [inter_sort[j][0],inter_sort[j][1]] not in tracker:
			tracker.append([inter_sort[j][0],inter_sort[j][1]])
			inter_final.append(inter_sort[j])
			hits.update({key: 0})

		if float(inter_sort[j][-1])/float(vline) > float(filt):
			count = hits.get(key)+1
			hits.update({key : count})

	print('-----Outputting scores-----')
	with open(f'{outdir}score.txt','w') as outfile:
		outfile.write('PDB   Pock      Target Vol   Pock Vol      Int Vol   Int %  # hits\n')
		for i in range(len(inter_final)):
			a=inter_final[i][0]
			p=inter_final[i][1]
			b=hits.get(f'{a} {p}')
			v=vols.get(f'{a} {p}')
			c=float(inter_final[i][-1])/float(vline)
			d=inter_final[i][-1]
			out=f"{a:<6}{p:<11}{vline:{12}.{8}}{v:{12}.{8}} {d:{9}.{7}} {c:{6}.{3}}    {b}\n"
			outfile.write(out)
"""
	s = np.genfromtxt(f'{outdir}score.txt', dtype='unicode', skip_header=1)

	if os.path.exists(f'{outdir}winners/master_scorefile.txt'):
		mode='a'
	else:
		mode='w'
		if not os.path.exists(f'{outdir}winners/'):
			os.mkdir(f'{outdir}winners/')
	
	with open(f'{outdir}winners/master_scorefile.txt', mode) as f:
		if mode == 'w':
			f.write('PDB   Pock      Target Vol   Pock Vol    Int Vol   Int %  #hits\n')
		if s.ndim == 1:
			pdb=f'{s[0]:<6}'
			pock=f'{s[1]:<7}'
			v1=f'{s[2]:{12}.{8}}'
			v2=f'{s[3]:{12}.{8}}'
			d=f'{s[4]:{11}.{7}}'
			dp=f'{s[5]:{7}.{3}}'
			h=f'{s[6]:<6}'
			f.write(pdb+pock+v1+v2+d+dp+h+'\n')	
		else:
			for i in range(len(s)):
				pdb=f'{s[i][0]:<6}'
				pock=f'{s[i][1]:<7}'
				v1=f'{s[i][2]:{12}.{8}}'
				v2=f'{s[i][3]:{12}.{8}}'
				d=f'{s[i][4]:{11}.{7}}'
				dp=f'{s[i][5]:{7}.{3}}'
				h=f'{s[i][6]:<6}'
				f.write(pdb+pock+v1+v2+d+dp+h+'\n')
"""

def rosetta_prep(outdir,indir,filt,hilt):
	if not os.path.exists(f'{outdir}rosetta'):
		os.mkdir(f'{outdir}rosetta')

	lst = np.genfromtxt(f'{outdir}score.txt', dtype='unicode', skip_header=1,
		usecols=(0,1,5,6))
	
	if lst.ndim == 1:
		array=[]
		pose_array=[]
		if float(lst[2]) > float(filt):
			if float(lst[3]) > float(hilt):
				a = lst[0]
				b = [ int(x) for x in lst[1] if x.isdigit() ]
				c=''
				for j in range(len(b)):
					c+=f'{b[j]}'
				fil = f'{indir}{a}_out/pockets/pocket{c}_atm.pdb'
				pose_array = np.genfromtxt(fil, dtype='unicode', skip_header=20,
					skip_footer=2, usecols=5)
				
				for i in range(len(pose_array)):
					if pose_array[i] not in array:
						if "." in pose_array[i]:
							continue
						else:
							array.append(pose_array[i])

				array = [ int(x) for x in array ]
				with open(f'{outdir}rosetta/{a}_pock{c}.pos','w') as outfile:
					for line in sorted(array):
						if line == sorted(array)[-1]:
							outfile.write(f'{line}')
						else:
							outfile.write(f'{line} ')

	else:
		for i in range(len(lst)):
			array=[]
			pose_array=[]

			if float(lst[i][2]) > float(filt):
				if float(lst[i][3]) > float(hilt):
					a = lst[i][0]
					b = [ int(x) for x in lst[i][1] if x.isdigit() ]
					c = ''
					for j in range(len(b)):
						c+=f'{b[j]}'
					fil = f'{indir}{a}_out/pockets/pocket{c}_atm.pdb'
					pose_array = np.genfromtxt(fil, dtype='unicode', skip_header=20,
						skip_footer=2, usecols=5)


					for i in range(len(pose_array)):
						if pose_array[i] not in array:
							if "." in pose_array[i]:
								continue
							else:
								array.append(pose_array[i])

					array = [ int(x) for x in array ]
					with open(f'{outdir}rosetta/{a}_pock{c}.pos','w') as outfile:
						for line in sorted(array):
							if line == sorted(array)[-1]:
								outfile.write(f'{line}')
							else:
								outfile.write(f'{line} ')

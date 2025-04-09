#!/usr/bin/env python
import glob, os, shutil, subprocess
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from typing import List, Tuple, Union
from aliases import fpocket, surf, vasp
from coordinate_manipulation import Coordinates

PathLike = Union[str, Path]

def format_pdbs(directory: PathLike) -> None:
        """
        Reformats file format for all files in 
        specified directory.
        Inputs:
                directory - directory to perform reformatting
        Outputs:
                returns None, all .ent files converted to .pdb
        """
        for f in directory.glob('*'):
            if f.suffix == '.ent':
                new_f = f.with_suffix('.pdb')
                os.rename(f, new_f)

def get_info(pdbpath: PathLike) -> None:
    """
    Obtain relevant information for each structure (protein name,
    experimental method, resolution, any cofactors).
    Inputs:
        pdbpath - full path of protein to extract info from
    Outputs:
        returns None, writes info to file in pdb directory
    """
    directory = pdbpath.parent
    structure = pdbpath.name

    # read file
    reader = [line for line in open(str(pdbpath)).readlines()]
    print(structure)
    
    # obtain title
    for line in reader:
        if 'COMPND   2' in line:
            title = line[20:-1].strip()
    
    # obtain experimental info
    expArr = []
    for line in reader:
        if 'EXPDTA' in line:
            expArr.append(line.replace(';',','))
        if 'RESOLUTION.' in line:
            expArr.append(line)

    if len(expArr) > 1:
        method = ' '.join(expArr[0].split()[1:])
        resolution = ' '.join(expArr[1].split()[-2:])
        exp_info = f'{method.strip()}: {resolution.strip()}'
    else:
        exp_info = 'NONE'

    # obtain cofactor info
    coflines = []
    for line in reader:
        if 'HETNAM' == line[:6]: 
            coflines.append(line.split()[1:])

    if coflines:
        cofs = []
        idx = 0
        cofs.append([coflines[0][0], ' '.join(coflines[0][1:])])

        for cofline in coflines[1:]:
            if cofline[1] == cofs[idx][0]:
                cofs[idx][1] = cofs[idx][1] + ''.join(cofline[2:])
            else:
                cofs.append([cofline[0], ' '.join(cofline[1:])])
                idx += 1

        cofactors = ';'.join([': '.join(cof) for cof in cofs])
    
    else:
        cofactors = 'NONE'

    # write out all info to file
    info_path = directory / 'infofiles'
    info_path.mkdir(exist_ok=True)

    outfile = f'{directory}/infofiles/{structure[:-4]}.info'
    with open(outfile, 'w') as out:
        out.write(f'{title}\n')
        out.write(f'{exp_info.strip()}\n')
        out.write(f'{cofactors.strip()}')                                 


def clean(structure: PathLike) -> None: 
    """
    PDBs straight from RCSB have a lot of information we don't care about.
    We need to isolate chain A and also filter out any alternative residues.
    It's ok to remove alt residues since we will be doing mutagenesis down the line.
    Finally for compatibility with downstream programs, atoms will be renumbered,
    starting from 1 with a simple mapping scheme.
    Inputs:
            structure - the particular structure to be cleaned, including filepath
    Outputs:
            returns None; writes out cleaned pdb over the old pdb file
    """

    # get lines of atoms
    lines = [line for line in open(f"{structure}").readlines() if line.split()[0] == 'ATOM']
    # filter out all chains but chain A and also remove alternative residues
    filtered = [line for line in lines if line[21] == 'A' and line[22:26].strip().isdigit()]
    # only take highest occupancy atoms
    highest = [f'{line[:16]} {line[17:]}' for line in filtered if line[16] in [' ', 'A']]

    # renumber atoms beginning from 1
    # get all the resids for each line
    resid = np.array([line[22:26].strip() for line in highest]).astype(int)

    # need to map to new values
    uniqueValues = np.unique(resid)
    # perform mapping
    renumbered = [np.where(uniqueValues == x)[0][0] + 1 for x in resid]
    
    # put the new numbering back into each line, including TER line
    final = [f'{line[:22]}{renumbered[i]:>4}{line[26:]}' for i, line in enumerate(highest)]
    final = final + ['TER']

    # move original pdb file
    (dirname, filename) = os.path.split(structure)
    fpath = f'{dirname}/original_pdbs/'
    if not os.path.exists(fpath):
        os.mkdir(fpath)

    shutil.move(structure, f'{fpath}{filename}')

    with open(structure,'w') as outfile:
        for line in final:
            outfile.write(line)

    outfile.close()

def write_pockets(directory: PathLike, pock: str, maximum: int) -> None:
        """
        Writes out pocket files according within the specified cutoffs.
        fpocket already filtered out any pockets with less than 80 dummy atoms,
        now pockets with over the max cutoff are excluded and all else are
        written into new pdbs.
        Inputs:
                directory - filepath of pdb file to test
                pock - pdb file to test for viable pockets
                maximum - cutoff filter for max dummy atoms
        Outputs:
                returns None, writes out files to directory
        """
        
        infile = open(f'{directory}{pock[:-4]}/{pock}','r')
        pock_array = [line for line in infile if "STP" in line]
        pocketID = [int(line[22:26].strip()) for line in pock_array]

        uniq = np.unique(np.array(pocketID))
        for i in range(uniq.shape[0]):
            j=i+1
            a = pocketID.count(j)
            if a < maximum:
                outfile = open(f'{directory}{pock.split("_")[0]}.pocket{j}.pdb','w')
                [outfile.write(line) for line in pock_array if int(line[22:26].strip()) == uniq[i]]

def identify_cofactors(directory: PathLike) -> None:
        """
        Iterate through each generated pocket that passes filters to identify
        which, if any, cofactors would have inhabited the pocket. This is
        accomplished via a simple shortest distance algorithm on the pocket prior
        to centering and alignment.
        Inputs:
                directory - filepath to generated pockets
        Outputs:
                returns None, rewrites each .info file to reflect pocket matching each
                        cofactor
        """
        
        # generate a list of every pocket, skips over aligned pockets if this is
        # a restart run due to aligned pockets lacking the .pdb extension
        all_pockets = [p for p in directory.glob('*pocket*.pdb')]

        # go through each pocket to determine if it inhabits a cofactor's space
        for pocket in all_pockets:
            base = os.path.basename(pocket)
            pnum = base.split('.')[1]

            print(base)
            # location and name of corresponding infofile
            infodir = f'{directory}infofiles/'
            infofile = f'{infodir}{base[:4]}.info'
            
            # read whole infofile, specifically take the cofactor line for now
            infolines = [line for line in open(infofile).readlines()]
            cofactline = infolines[-1]
            
            if cofactline != 'NONE':
                # cofactor format is - 3 letter code: full name: pocket#/'Not present'; .....
                cofactors = [c.split(':')[0].strip() for c in cofactline.split(';')]

                # get pocket coordinates
                c = [[l[30:38].strip(),l[38:46].strip(),l[46:54].strip()] for l in open(pocket).readlines()]

                # extract all heteroatom coordinate lines from original pdb
                cof = [line for line in open(f'{directory}original_pdbs/{base[:4]}.pdb').readlines() if line[:6] == 'HETATM']

                # generate array where 0 = cofactor not in pocket, 1 = cofactor in pocket
                # initially all cofactors set to 0
                cof_present = np.array([[c,0] for c in cofactors])
                for i, cofactor in enumerate(cofactors):
                    # get specific cofactor coordinates
                    c2 = [[l[30:38].strip(),l[38:46].strip(),l[46:54].strip()] for l in cof if l[17:20].strip() == cofactor]

                    # measure minimum pairwise euclidean distance of cofactor and pocket coords
                    d = np.min(np.min(cdist(np.array(c).astype(float), np.array(c2).astype(float), 'euclidean')))

                    # cofactor occupacy of pocket defined as any pair of atoms <1 angstrom, set array to 1
                    if d < 1:
                        cof_present[np.where(cof_present == cofactor)[0][0],1] = 1

                # identified array tracks any changes to make to infofile
                identified = []
                for i, cofactor in enumerate(cofactline.split(';')):
                    # info contains just the info for the current cofactor
                    info = [c.strip() for c in cofactor.split(':')]

                    # if the entry for this cofactor in cof_present is '1', it occupies this pocket
                    if cof_present[np.where(cof_present == info[0])[0][0],1] == '1':
                            # if this cofactor HAS NOT been assigned a pocket, assign this one
                            if info[-1] == 'Not present':
                                info[-1] = pnum
                            # if this cofactor HAS been assigned a pocket, append this one
                            else:
                                info[-1] = ', '.join([info[-1],pnum])

                    identified.append(info)

                # generate the updated cofactor info line
                new_cofline = []
                for i in range(len(identified)):
                    cur = ': '.join(identified[i])
                    if i > 0:
                        new_cofline = '; '.join([new_cofline, cur])
                    else:
                        new_cofline = cur

                # write out updated infofile
                with open(infofile,'w') as out:
                    out.write(f'{infolines[0]}')
                    out.write(f'{infolines[1]}')
                    out.write(f'{new_cofline}')


def principal(array: np.ndarray) -> np.ndarray:
        """
        Find the principal axis in a point cloud.
        Inputs:
                array - points to find principal axis of
        Outputs:
                axis1 - the principal axis of the points        
        """

        inertia = array.T @ array
        e_values, e_vectors = np.linalg.eig(inertia)
        order = np.argsort(e_values)
        eval3,eval2,eval1 = e_values[order]
        axis3,axis2,axis1 = e_vectors[:,order].T
        return axis1


def align(array: np.ndarray, 
          a: np.ndarray) -> np.ndarray:
        """
        Align a point cloud with its principal axis along
        the z-axis. This is done according to the formula:
        R = I + vx + vx^2/(1+c) where I is the identity 3x3,
        vx is a skew-symmetric of a1 x a2, and c is the dot 
        product a1*a2.
        Inputs:
                array - point cloud to align
                a - vector corresponding to principal axis
        Outputs:
                aligned - point cloud that has been aligned
        """

        aligned=np.zeros((array.shape[0],array.shape[1])) 
        b = [0,0,1] 
        v = np.cross(a,b) 
        c = np.dot(a,b) 
        I = np.eye(3,3)

        vx = np.array(
            [
                [0,    -v[2],  v[1]],
                [v[2],     0, -v[0]],
                [-v[1], v[0],     0]
            ]
        )
        R = I + vx + (vx @ vx)/(1+c)
        aligned = R @ array.T

        return aligned.T        


def center(array: np.ndarray) -> np.ndarray:
        """
        Moves a point cloud so the geometric center is at the
        origin (0,0,0). Accomplished by subtracting each point
        by the geometric center of mass of the point cloud.
        Inputs:
                array - points to recenter
        Outputs:
                centered - array of recentered points
        """

        centered = np.zeros((array.shape[0], array.shape[1]))
        com = array.mean(0)

        for i in range(array.shape[0]):
            centered[i] = array[i] - com

        return centered

def find_pockets(indir: PathLike, 
                 alphas: int) -> None:
        """
        Run fpocket for all pdbs in a given directory.
        Inputs:
                indir - directory containing structures to
                                run fpocket on
                alphas - minimum alpha sphere cutoff, this
                                is an option in fpocket
        Outputs:
                returns None, just runs fpocket and prints
                        whether output directories contain files
        """

        for pdb in indir.glob('*.pdb'):
            root = pdb.name.split('.')[0]
            if not (indir / f'{root}_out').exists:
                print(f'---RUNNING fpocket ON {root}---')
                args = (fpocket, '-f', pdb, '-i', str(alphas))
                subprocess.run(args)

def gen_surfs(outdir: PathLike, 
              indir: PathLike, 
              pocket: str) -> None:
        """
        Run SURF to generate surf file of given pocket.
        Inputs:
                outdir - directory to output surf file to
                indir - filepath to pocket
                pocket - pdb ID and pocket number
        Outputs:
                returns None, runs SURF and outputs logfile
        """

        pock = indir / f'aligned.{pocket}'
        print(pocket)
        n, p = pocket.split('.')[:2]

        arg = (surf, '-surfProbeGen', pock, outdir / 'VASP' / 'pockets' / f'{n}_{p}.SURF', '3', '.5')
        out_name = outdir / 'VASP' / 'pockets' / 'surf.log'
        out = open(str(outname), 'w')

        subprocess.run(arg, stdout=out)
        out.close()


def intersect(outdir: PathLike, 
              initial: PathLike, 
              pocket: str, 
              snum: int, 
              total: int, 
              samples: int, 
              catalogue: List[str], 
              full: bool = True) -> None:
        """
        Run intersect VASP on given pocket.

        Inputs:
                outdir - output directory for VASP runs
                initial - directory containing target structures
                                that will be intersected with given pocket
                pocket - pocket structure to be tested against target
                snum - structure number, for progress display
                total - total number of structures, for progress display
                samples - number of conformations being sampled for display
                catalogue - list of target structures to intersect
                full - switch for the inclusion of translational sampling
        
        Outputs:
                None, runs VASP for user-specified catalogue
        """
        count = 0
        struc = f'{outdir}VASP/pockets/{pocket.split(".")[0]}_{pocket.split(".")[1]}.SURF'
        n = f"{pocket.split('.')[0]}_{pocket.split('.')[1]}"

        # clean up outputs so that you can perform operations in
        # the VASP directory, otherwise too many files
        if not os.path.exists(f'{outdir}VASP/{n}/'):
            os.mkdir(f'{outdir}VASP/{n}/')

        for init in catalogue:
            count += 1
            if full:
                modulo = 25
            else:
                modulo = 5

            if (count%500==0):
                print(f'-----{snum+1}/{total} pockets to be sampled-----')

            if (count%modulo==0):
                print(f'{pocket}: {count}/{samples} conformations')

            fname = os.path.basename(init).split('.')[:5]
            surf_file = outdir / 'VASP' / str(n) / f'intersect.{n}.{".".join(fname)}.SURF'
            arg = (vasp, '-csg', struc, init, 'I', surf_file, '.5')
            out = open(out / 'intersect.log','w')
            subprocess.run(arg, stdout=out)
            out.close()

        arg = (surf, '-surveyVolume', struc)
        out = open(outdir / 'VASP' / 'pockets' / f'{n}.vol.txt', 'w')
        subprocess.run(arg, stdout=out)
        out.close()

def gen_short_sample(targetDir: PathLike) -> np.ndarray:
        """
        Generate the short sampling array.
        Inputs:
                targetDir - directory containing target structures
        Outputs:
                short - array of target structures that comprise the
                                short screen
        """

        short = targetDir.glob('*conf0*.0.0.SURF')
        translations = targetDir.glob('*conf0.0.0.*.0.SURF')
        return np.unique(np.append(short, translations))

def gen_long_sample(initialdir: PathLike, 
                    short: np.ndarray, 
                    _result: bool, 
                    _tSamp: List[bool]) -> np.ndarray:
        """
        Generate the long sampling array. This is done by subtracting out
        the short array from the totality of structures. Also, the translational
        screen is applied to exclude any translations where necessary.
        Inputs:
                initialdir - directory containing target structures
                short - array of short screen structures
                _result - continue non-translated screening
                _tSamp - boolean array of translational sampling results
        Outpus:
                full - array of structures that comprise the full-length screen
        """
        
        if _result:
            # remove the short sample from the full list
            full = np.setdiff1d(initialdir.glob('*.SURF'), short)

            # remove appropriate translations from the full list
            for i, trans in enumerate(_tSamp):
                if not trans:
                    translation = initialdir.glob('*conf*.*.*.{i+1}.*.SURF')
                    full = np.setdiff1d(full, translation)

        else: # this means the non-translated screen failed
            full = np.array([])
            for i, trans in enumerate(_tSamp):
                if trans:
                    translation = initialdir.glob('*conf*.*.*.{i+1}.*.SURF')
                    full = np.append(full, translation)

        return full

def extract_score(outdir: PathLike, 
                  pocket: str) -> None:
        """
        Runs SURF to get the intersect score for all intersections performed
        on supplied pocket.
        Inputs:
                outdir - output directory where the scores will be put
                pocket - the pocket to obtain scores for
        Outputs
                returns None, runs SURF -surveyVolume to generate scores
        """
        
        name = '_'.join(pocket.split('.')[:2])
        for inter in (outdir / 'VASP' / name).glob('*.SURF'):
            p = '_'.join(inter.name.split('.')[1:7])
            arg = (surf, '-surveyVolume', inter)
            out = open(outdir / 'VASP' / name / f'{p}_iscore.txt', 'w')
            subprocess.run(arg, stdout=out)
            out.close()


def screen_check(outdir: PathLike, 
                 pocket: str, 
                 cut: float, 
                 _tSamp: List[bool]) -> Tuple[bool, List[bool]]:
        """
        Check if a short screen was successful. Additionally, map out
        which translations to perform in full sampling.
        Inputs:
                outdir - output directory filepath for VASP scores
                pocket - which pocket to check screen on
                cut - volume cutoff for screening
                _tSamp = copy of tSamp boolean array, dictates trans sampling
        Outputs:
                result - whether short screen was successful or not
                _tSamp = updated copy of tSamp
        """
        result = False
        name = '_'.join(pocket.split('.')[:2])

        # check small screen first
        sText = outdir / 'VASP' / name
        values = np.array(
            [
                float([lines.split() for lines in open(struc, 'r')][-1][-1])
                for struc in sText.glob(f'{name}_conf0*_0_0_iscore.txt')
            ]
        )

        if np.where(values > cut)[0].size > 0:
                result = True

        for i in range(len(_tSamp)):
            scorefile = outdir / 'VASP' / name / f'{name}_conf0_0_0_{i+1}_0_iscore.txt'
            tInt = [line.split() 
                    for line in open(scorefile).readlines()][-1][-1]
            _tSamp[i] = float(tInt) > cut

        return result, _tSamp


def original_volume(outdir: PathLike, 
                    p: str) -> None:
        """
        Runs SURF -surveyVolume on supplied pocket.
        Inputs:
                outdir - directory where to output volume
                p - pocket to get original volume of
        Outputs:
                returns None, generates volume file for pocket
        """

        v = outdir / 'VASP' / 'pockets' / f'{"_".join(p.split(".")[:2])}.SURF'
        n = v.name.split('.')[0]
        arg = (surf, '-surveyVolume', v)
        out = open(outdir / 'VASP' / 'pockets' / f'{n}.vol.txt', 'w')
        subprocess.run(arg, stdout=out)
        out.close()


def gen_scorefile(outdir: PathLike) -> bool:
    """
    Generates new scorefile or returns False if one exists. If a scorefile exists the main
    function should throw an error as you may be overwriting previous data.
    Inputs:
        outdir - output directory containing filepath for scores and scorefile
    Returns:
        bool - True if file written, False if it already exists
    """
    filepath = outdir / 'score.txt'
    if filepath.exists:
        return False

    else:
        with open(filepath, 'w') as sfile:
            header = 'PDB   Pock     Target Vol  Pock Vol   Int Vol  Int%  #hits'
            header += '       Exp. Method                       Cofactor(s)'
            header += '                 Protein Name\n'
            sfile.write(header)

        sfile.close()
        return True
     

def append_scorefile(outdir: PathLike, 
                     pdbdir: PathLike, 
                     struc: str, 
                     filt: float, 
                     vol: float) -> None:
        """
        Appends run to scorefile. Checks if structure has been written to file before
        and outputs updated values if so. 
        Inputs:
                outdir - output directory containing filepath for scores and scorefile
                pdbdir - directory containing original pdbs and infofiles
                struc - pocket to append score to file
                filt - int% filter, used to count number of hits
                vol - volume of target pocket
        Outputs:
                returns None, appends values to scorefile
        """

        print(struc)    
        pdb, pock = struc.split('.')[:2]

        # get pocket volume from vol file
        vasp = outdir / 'VASP'
        f = open(VASP / 'pockets' / f'{pdb}_{pock}.vol.txt').readlines()[-1]
        v = float(f.split()[-1])

        # list of all scorefiles to extract data from
        scores = [s for s in (vasp / f'{pdb}_{pock}').glob('*_iscore.txt')]

        hitCounter = 0
        bestScore = 0
        curScore = 0
        
        print(f'-----Getting {pdb} {pock} Scores-----')
        # iterate through all scores and track hits and best int%
        for score in scores:
            curScore = float(open(score).readlines()[-1].split()[-1])
            if not bestScore or curScore > bestScore:
                bestScore = curScore
            if curScore / vol > filt:
                hitCounter += 1
            curScore = 0

        print('-----Updating Scorefile-----')

        # get exp. method, cofactor and protein name information
        info = [line for line in open(pdbdir / 'infofiles' / f'{pdb}.info').readlines()]
        
        cofactor = None
        for cof in info[-1].split(';'):
            if cof.split(':')[-1].strip() == pock:
                cofactor = ':'.join(cof.split(':')[:2])
        
        l = [pdb, pock, vol, v, bestScore, float(bestScore / vol), hitCounter]
        l += [info[1][:-2], cofactor, info[0][:-2]]
        l = [str(x) for x in l]
        
        with open(outdir / 'score.txt', 'a') as sfile:
            sfile.write(';'.join(l) + '\n')

def delete_surfs(structure: PathLike, 
                 outdir: PathLike) -> None:
        """
        In an effort to keep the data footprint small, delete surf files. this
        should only be called after scorefile has been written out.
        Inputs:
                structure - name of structure to delete files for
                outdir - directory in filepath of surf files
        Outputs:
                returns None, deletes appropriate files
        """
        
        print(f'-----Cleaning up {structure} SURF/VASP files-----')
        name = '_'.join(structure.split('.')[:2])
        vasp = outdir / 'VASP'
        shutil.rmtree(vasp / name)
        os.remove(vasp / 'pockets' / f'{name}.SURF')
        os.remove(vasp / 'pockets' / f'{name}.vol.txt')


def move_scored_structures(outdir: PathLike, 
                           pdbdir: PathLike) -> None:
    """
    Function to move cleaned PDBs of structures that make it to the
    scorefile to an output directory.
    Inputs:
            outdir - output directory where scorefile is, also in the
                            filepath of the PDB output directory
            pdbdir - directory where PDBs can be found initially
    Outputs:
            returns None, moves PDBs to output directory
    """
    
    score = open(outdir / 'score.txt', 'r')
    raw = [[ele.strip() for ele in line.split(';')[:2]] for line in score.readlines()]
    scored = ['.'.join(row) for row in raw]

    if not isinstance(raw[0], list):
        lst = lst.reshape((1,2))
        
    path = outdir / 'scored_pdbs'
    path.mkdir(exist_ok=True)

    # if pdb still in pdbdir and not in scored pdbs, move it there
    for pdb in scored:
        if (pdbdir / f'{pdb}.pdb').exists():
            if not (path / f'{pdb}.pdb').exists():
                shutil.move(pdbdir / f'{pdb}.pdb', path)

def rosetta_prep(outdir: PathLike, 
                 indir: PathLike, 
                 filt: float, 
                 hilt: int, 
                 pocket: str) -> None:
    """
    This function generates rosetta .pos files for each structure that
    passes both the int% and hit filters. The scorefile is read and for each
    success the fpocket output for that pocket is read and the resID
    is acquired for each residue that lines the pocket.
    Inputs:
            outdir - output directory where scorefile various outputs are
            indir - the pdb directory, also containing fpocket outputs
            filt - int% filter
            hilt - hit filter
            pocket - pdb ID and pocket number of each structure
    """

    # check if the rosetta output directoyr exists, else make it
    rdir = outdir / 'rosetta'
    rdir.mkdir(exist_ok=True)
    
    # get our list of pose files to make from the scorefile
    s = open(outdir / 'score.txt', 'r')
    _ = s.readline()
            
    raw = [[ele.strip() for ele in line.split(';')] for line in s.readlines()]
    s.close()

    lst = np.array(raw, dtype=str)

    if not isinstance(raw[0], list):
        lst = lst.reshape((1,10))
    
    for i in range(lst.shape[0]):
        array=[]
        pose_array=[]

        if all([float(lst[i][5]) > filt, int(lst[i][6]) > hilt]):
            a = lst[i][0]
            b = ''.join([ x for x in lst[i][1] if x.isdigit() ])
            fil = indir / f'{a}_out' / 'pockets' / f'pocket{b}_atm.pdb'
            
            lines = []
            with open(fil, 'r') as prefile:
                for i, line in enumerate(prefile):
                    if line[:4] == 'ATOM':
                        lines.append(line.split()[:6])

            lines = np.asarray(lines)
            pose_array = [resid for resid in lines[:, -1]]

            for i in range(len(pose_array)):
                if pose_array[i] not in array:
                    if "." in pose_array[i]:
                        continue
                    else:
                        array.append(pose_array[i])

            array = [ int(x) if x[-1].isdigit() else int(x[:-1]) for x in array ]
            with open(rdir / f'{a}_pock{b}.pos', 'w') as outfile:
                for line in sorted(array):
                    outline = ' '.join([str(x) for x in array])
                    outfile.write(outline)

def restart_run(outputdir: PathLike, 
                pdbdir: PathLike) -> List[str]:
    """
    Read in checkpoint file to restart a run.
    Inputs:
        outputdir - checkpoint filepath
        pdbdir - directory where scaffold pdbs are stored
    Returns:
        list of structures to run on
    """

    f = open(outputdir / 'checkpoint.chk', 'r')
    complete = [line.strip() for line in f.readlines()]
    f.close()

    total = [pdb.name for pdb in pdbdir.glob('*pocket*pdb')]

    return [b[:-4] for b in total if not all(a in b for a in complete)]


def update_checkpoint(outputdir: PathLike, 
                      completed_structure: str) -> None:
    """
    Updates the checkpoint file with `pdb` and `pock` in the csv format.
    Inputs:
        outputdir - 
        completed_structure -
    Returns:
        None
    """
    f = open(outputdir / 'checkpoint.chk', 'a')
    f.write(f'{completed_structure}\n')
    f.close()

def preprocess(checkpoint: bool, 
               pdbdir: PathLike, 
               targetdir: PathLike, 
               alpha: float, 
               cutoff: float, 
               outputdir: str) -> Tuple[List[str], int, int, 
                                        List[str], float]:
    """
    Preprocessing workflow. Setups up checkpointing and obtains list of
    structures to run pocketSearch on.
    Inputs:
        checkpoint -
        pdbdir -
        targetdir -
        alpha -
        cutoff -
        outputdir -
    Outputs:
        tracker -
        t -
        s -
        short_sample -
        vol -
        chkpt -
    """
    # read in checkpoint; else do pocketSearch prepwork
    if checkpoint:
        tracker = restart_run(outputdir, pdbdir)
    
    else:
        # rename everything to be consistent with typical naming schemes
        format_pdbs(pdbdir)
        
        # clean up pdbs, removing cofactors and renumbering where necessary
        (pdbdir / 'original_pdbs').mkdir(exist_ok=True)
            
        for unclean in pdbdir.glob('*.pdb'):
            get_info(unclean) 
            clean(unclean)

        # get target pocket alpha sphere count for fpocket cutoff calculation
        for target in targetdir.glob('*pocket*'):
            aspheres = len(open(target).readlines()) + 1
    
        fpocket_min = int(aspheres * alpha)
        fpocket_max = int(aspheres * cutoff)
    
        # generate pockets
        find_pockets(pdbdir, fpocket_min)
        
        name_array = [pdb.name for pdb in pdbdir.glob('*_out/*_out.pdb')]
    
        print('Writing out pockets...') 
        for entry in name_array:
            print(f'Extracting pockets from {entry}')
            write_pockets(pdbdir, entry, fpocket_max)
        
        # identify cofactors that match each pocket that passes filter
        identify_cofactors(pdbdir)
    
        print('Processing pockets...')
        # center and align pockets
        prealigned = []
        for pdb in pdbdir.glob('*.pocket*.pdb'):
            # check if original pdb still there, if not it has been scored already
            # and this structure should be skipped
            stem = pdb.name.split('.')
            if (pdbdir / f'{stem[0]}.pdb').exists():
                prealigned.append([stem[0], stem[1]])
    
        tracker = []
        for entry in prealigned:
            print(entry)
            tracker.append(f'{entry[0]}.{entry[1]}')
            coordsystem = Coordinates(pdbdir, entry[0], pnum=entry[1])
            coordsystem.get_coords()
            coordsystem.center()
            coordsystem.principal()
            aligned = coordsystem.align()
            coordsystem.make_pdb(aligned)
        
        # generate surf files and run VASP on pockets
        # check to see if VASP scores and VASP pockets directory exists
        vasp = outputdir / 'VASP'
        vasp.mkdir(exist_ok=True)
        (vasp / 'scores').mkdir(exist_ok=True)
        (vasp / 'pockets').mkdir(exist_ok=True)
        
        # write out checkpoint file
        # NOTE: WTF is this?
        chkpt = outputdir / 'checkpoint.chk'
        open(chkpt, 'a').close()

        print('Preprocessing complete...')
    
    
    # conformational sampling for initial screen
    short_sample = gen_short_sample(targetdir)
    s = len(short_sample)
    t = len(tracker)
    
    # target pocket volume
    vol = float(open(targetdir / 'vol.txt','r').readlines()[-1].split()[-1])

    return tracker, t, s, short_sample, vol

def pocket_search(i: int, 
                  structure: str, 
                  outputdir: PathLike, 
                  pdbdir: PathLike, 
                  targetdir: PathLike, 
                  t: int,
                  s: int, 
                  short_sample: List[str], 
                  min_intersect:float, 
                  vol: float, 
                  screen: float, 
                  min_hits: int) -> None:
    """
    The main pocketSearch flow control function. Takes singular elements of the
    total pocketSearch geometric screen and generates surface files, obtains the
    intersection surface with the target and scores all the intersections.
    Inputs:
        i
        structure
        outputdir
        pdbdir
        targetdir
        t
        s
        short_sample
        min_intersect
        vol
        screen
        min_hits
        chkpt
    Outputs:
        None
    """
    # dummy variables to guide conformational sampling
    tSamp = np.full(6, True)
    print(f'-----Running on: {structure}-----')
    
    # get surf file
    gen_surfs(outputdir, pdbdir, structure)
    
    # run short screen of intersect VASP on each structure
    intersect(outputdir, targetdir, structure, i, t, s, short_sample, full=False)
    
    # get scores and update scorefile
    extract_score(outputdir, structure)
    original_volume(outputdir, structure)
    
    result, tSamp = screen_check(outputdir, structure, screen*vol, tSamp)

    # only perform full intersect sampling if initial screen passed,
    # also performs only translations in directions that pass Samp
    if np.any(np.append(result, tSamp)):
        print(f'-----Full Screen on: {structure}-----')
        long_sample = gen_long_sample(targetdir, short_sample, result, tSamp)
        intersect(outputdir, targetdir, structure, i, t, len(long_sample), long_sample)

        # extract each score
        extract_score(outputdir, structure)

    # append scorefile
    append_scorefile(outputdir, pdbdir, structure, min_intersect, vol)

    # prep for ROSETTA
    rosetta_prep(outputdir, pdbdir, min_intersect, min_hits, structure)

    # move scored structures
    #move_scored_structures(outputdir, pdbdir)
    
    # update checkpoint file
    update_checkpoint(outputdir, structure)

    # remove surf/vasp files. they take up an enormous amount of storage
    # otherwise (~500Mb per structure ran)
    delete_surfs(structure, outputdir)

def postprocessing(outdir: PathLike) -> None:
    """
    Cleans up scorefile so that it is easier to read.
    Inputs:
        outdir - output directory where scorefile is located
    Outputs:
        None
    """
    score = outdir / 'score.txt'
    sfile = open(score, 'r')
    header = sfile.readline()
    contents = [line.split(';') for line in sfile.readlines()]
    sfile.close()
    
    backup = score.with_suffix('.BAK')
    shutil.move(score, backup)

    with open(score, 'w') as outfile:
        outfile.write(header)
        for line in contents:
            pdb, pock, tv, pv, iv, ip, nh, method, cof, name = line
            oline = f'{pdb:<6}{pock:<9}{tv:<11.8}{pv:<11.8}{iv:<9.7}{ip:<7.5}'
            oline += f'{nh:>6.4}  {method:<38}{cof}  {name:>17}'

            outfile.write(oline)

    os.remove(backup)

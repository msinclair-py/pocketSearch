#!/usr/bin/env python
import glob, os, shutil, subprocess
import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple
from aliases import fpocket
from coordinate_manipulation import Coordinates
import ripser
import persim

def check_format(directory: str) -> str:
        """
        Reformats directory name for downstream consistency.
        Inputs:
                directory - directory to be checked
        Outputs:
                returns directory with last character '/'
        """

        if directory[-1] != '/':
                directory = f'{directory}/'
        if not os.path.exists(directory):
                print('Directory doesn\'t exist')
        else:
                print('Directory exists')
        return directory


def format_pdbs(directory: str) -> None:
        """
        Reformats file format for all files in 
        specified directory.
        Inputs:
                directory - directory to perform reformatting
        Outputs:
                returns None, all .ent files converted to .pdb
        """

        for f in glob.glob(f'{directory}/*'):
                old = os.path.basename(f)
                if old[-4:] == '.ent':
                        new = f'{old[3:-4]}.pdb'
                        os.rename(f,f'{directory}{new}')


def get_info(pdbpath: str) -> None:
    """
    Obtain relevant information for each structure (protein name,
    experimental method, resolution, any cofactors).
    Inputs:
        pdbpath - full path of protein to extract info from
    Outputs:
        returns None, writes info to file in pdb directory
    """
    
    structure, directory = os.path.split(pdbpath)

    # read file
    reader = [line for line in open(pdbpath).readlines()]
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
    if not os.path.exists(f'{directory}infofiles/'):
        os.mkdir(f'{directory}infofiles/')

    outfile = f'{directory}infofiles/{structure[:-4]}.info'
    with open(outfile, 'w') as out:
        out.write(f'{title}\n')
        out.write(f'{exp_info.strip()}\n')
        out.write(f'{cofactors.strip()}')                                 


def clean(structure: str) -> None: 
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


def write_pockets(directory: str, pock: str, maximum: int) -> None:
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
        pocketID = [int(line.split()[5]) for line in pock_array]

        uniq = np.unique(np.array(pocketID))
        for i in range(uniq.shape[0]):
                j=i+1
                a = pocketID.count(j)
                if a < maximum:
                        outfile = open(f'{directory}{pock.split("_")[0]}.pocket{j}.pdb','w')
                        [outfile.write(line) for line in pock_array if int(line[22:26].strip()) == uniq[i]]


def identify_cofactors(directory: str) -> None:
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
        all_pockets = [ p for p in glob.glob(f'{directory}*pocket*.pdb') ]

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


def find_pockets(indir: str, alphas: int) -> None:
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

        for i in glob.glob(f'{indir}*.pdb'):
            root = os.path.basename(i).split('.')[0]
            if not os.path.exists(f'{indir}{root}_out/'):
                print(f'---RUNNING fpocket ON {root}---')
                args=(fpocket,'-f',i,'-i',alphas)
                str_args=[ str(x) for x in args ]
                subprocess.run(str_args)
        

#FIXME
def gen_scorefile(outdir: str) -> bool:
    """
    Generates new scorefile or returns False if one exists. If a scorefile exists the main
    function should throw an error as you may be overwriting previous data.
    Inputs:
        outdir - output directory containing filepath for scores and scorefile
    Returns:
        bool - True if file written, False if it already exists
    """
    filepath = f'{outdir}score.txt'
    if os.path.exists(filepath):
        return False

    else:
        with open(filepath, 'w') as sfile:
            header = 'PDB   Pock     Target Vol  Pock Vol   Int Vol  Int%  #hits'
            header += '       Exp. Method                       Cofactor(s)'
            header += '                 Protein Name\n'
            sfile.write(header)

        sfile.close()
        return True
     

#FIXME
def append_scorefile(outdir: str, pdbdir: str, struc: str, 
                   filt: float, vol: float) -> None:
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
        pdb = struc.split('.')[0]
        pock = struc.split('.')[1]

        # get pocket volume from vol file
        f = open(f'{outdir}VASP/pockets/{pdb}_{pock}.vol.txt').readlines()[-1]
        v = float(f.split()[-1])

        # list of all scorefiles to extract data from
        scores = [s for s in glob.glob(f'{outdir}VASP/{pdb}_{pock}/*_iscore.txt')]

        hitCounter = 0
        bestScore = 0
        curScore = 0
        
        print(f'-----Getting {pdb} {pock} Scores-----')
        # iterate through all scores and track hits and best int%
        for score in scores:
                curScore = float(open(score).readlines()[-1].split()[-1])
                if not bestScore or curScore > bestScore:
                        bestScore = curScore
                if curScore/vol > filt:
                        hitCounter+=1
                curScore = 0

        print('-----Updating Scorefile-----')

        # get exp. method, cofactor and protein name information
        info = [line for line in open(f'{pdbdir}infofiles/{pdb}.info').readlines()]
        
        cofactor = None
        for cof in info[-1].split(';'):
            if cof.split(':')[-1].strip() == pock:
                cofactor = ':'.join(cof.split(':')[:2])
        
        l = [pdb, pock, vol, v, bestScore, float(bestScore/vol), hitCounter]
        l += [info[1][:-2], cofactor, info[0][:-2]]
        l = [str(x) for x in l]
        
        with open(f'{outdir}score.txt','a') as sfile:
            sfile.write(';'.join(l) + '\n')


def move_scored_structures(outdir: str, pdbdir: str) -> None:
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
    
    score = open(f'{outdir}score.txt', 'r')
    raw = [[ele.strip() for ele in line.split(';')[:2]] for line in score.readlines()]
    scored = ['.'.join(row) for row in raw]

    if not isinstance(raw[0], list):
        lst = lst.reshape((1,2))
        
    path = f'{outdir}scored_pdbs/'

    # ensure scored pdbs directory exists
    if not os.path.exists(path):
        os.mkdir(path)

    # if pdb still in pdbdir and not in scored pdbs, move it there
    for pdb in scored:
        if os.path.exists(f'{pdbdir}{pdb}.pdb'):
            if not os.path.exists(f'{path}{pdb}.pdb'):
                shutil.move(f'{pdbdir}{pdb}.pdb', path)


#FIXME
def rosetta_prep(outdir: str, indir: str, filt: float, 
                   hilt: int, pocket: str) -> None:
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
    if not os.path.exists(f'{outdir}rosetta'):
            os.mkdir(f'{outdir}rosetta')
    
    # get our list of pose files to make from the scorefile
    s = open(f'{outdir}score.txt', 'r')
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
            fil = f'{indir}{a}_out/pockets/pocket{b}_atm.pdb'
            
            lines = []
            with open(fil, 'r') as prefile:
                for i, line in enumerate(prefile):
                    if line[:4] == 'ATOM':
                        lines.append(line.split()[:6])

            lines = np.asarray(lines)
            pose_array = [resid for resid in lines[:,-1]]

            for i in range(len(pose_array)):
                if pose_array[i] not in array:
                    if "." in pose_array[i]:
                        continue
                    else:
                        array.append(pose_array[i])

            array = [ int(x) if x[-1].isdigit() else int(x[:-1]) for x in array ]
            with open(f'{outdir}rosetta/{a}_pock{b}.pos','w') as outfile:
                for line in sorted(array):
                    outline = ' '.join([str(x) for x in array])
                    outfile.write(outline)


def restart_run(outputdir: str, pdbdir: str) -> List[str]:
    """
    Read in checkpoint file to restart a run.
    Inputs:
        outputdir - checkpoint filepath
        pdbdir - directory where scaffold pdbs are stored
    Returns:
        list of structures to run on
    """

    f = open(f'{outputdir}checkpoint.chk', 'r')
    complete = [line.strip() for line in f.readlines()]
    f.close()

    total = [os.path.basename(pdb) for pdb in glob.glob(f'{pdbdir}*pocket*pdb')]

    return [b[:-4] for b in total if all(a not in b for a in complete)]


def update_checkpoint(outputdir: str, completed_structure: str) -> None:
    """
    Updates the checkpoint file with `pdb` and `pock` in the csv format.
    Inputs:
        outputdir - 
        completed_structure -
    Returns:
        None
    """
    
    f = open(f'{outputdir}checkpoint.chk', 'a')
    f.write(f'{completed_structure}\n')
    f.close()


#FIXME
def preprocess(checkpoint: bool, pdbdir: str, targetdir: str, alpha: float, 
                cutoff: float, outputdir: str) -> List[str]:
    """
    Preprocessing workflow. Setups up checkpointing and obtains list of
    structures to run pocketSearch on.
    Inputs:
        checkpoint - Boolean value which indicates whether or not to look
                        for a checkpoint file to restart a run
        pdbdir - Path to where the pdb scaffolds are kept
        targetdir - Path to where the target pocket is located
        alpha - Min. percent of target pocket alpha spheres to be considered
                    for screening
        cutoff - Max. percent of target pocket alpha spheres to be considered
                    for screening
        outputdir - Path to where the outputs of screening will be stored
    Outputs:
        tracker - List of all pdbs that still need to be ran
    """

    # read in checkpoint; else do pocketSearch prepwork
    if checkpoint:
        tracker = restart_run(outputdir, pdbdir)
    
    else:
        # rename everything to be consistent with typical naming schemes
        format_pdbs(pdbdir)
        
        # clean up pdbs, removing cofactors and renumbering where necessary
        if not os.path.exists(f'{pdbdir}/original_pdbs/'):
            os.mkdir(f'{pdbdir}/original_pdbs/')
            
        for unclean in glob.glob(f'{pdbdir}*.pdb'):
            get_info(os.path.basename(unclean), pdbdir) 
            clean(unclean)

        # get target pocket alpha sphere count for fpocket cutoff calculation
        for target in glob.glob(f'{targetdir}*pocket*'):
            with open(target) as f:
                for i, l in enumerate(f):
                    pass
                aspheres = i + 1
    
        fpocket_min = int(aspheres * alpha)
        fpocket_max = int(aspheres * cutoff)
    
        # generate pockets
        find_pockets(pdbdir, fpocket_min)
        
        name_array=[os.path.basename(name) for name in glob.glob(f'{pdbdir}*_out/*_out.pdb')]
    
        print('Writing out pockets...') 
        for entry in name_array:
            print(f'Extracting pockets from {entry}')
            write_pockets(pdbdir, entry, fpocket_max)
        
        # identify cofactors that match each pocket that passes filter
        identify_cofactors(pdbdir)
    
        # write out checkpoint file
        chkpt = f'{outputdir}checkpoint.chk'
        open(chkpt, 'a').close()

        print('Preprocessing complete...')
    
    return tracker 


def get_homology_diagram(coords: List[float], outpath: str, name: str) -> None:
    """
    Given an input set of coordinates, generate a persistence homology diagram.
    This is done using the ripser package from scikit-tda
    Inputs:
        coords
    Outputs:
        None
    """
    diagram = ripser.ripser(coords, maxdim=2)['dgms']
    np.save(f'{outpath}/{name}.npy', diagram)


def obtain_wasserstein(diag1: np.ndarray, diag2: np.ndarray, 
                     outpath: str, name: str) -> None:
    """
    Takes two input persistence diagrams and computes the wasserstein
    distance between them. Writes this distance out to a file.
    Inputs:
        diag1
        diag2
        outpath
        name
    Outputs:
        None
    """
    wasserstein = [persim.wasserstein(d1, d2) for (d1, d2) in zip(diag1, diag2)]
    np.save(f'{outpath}/scores/{name}.npy', np.array(wasserstein))


#FIXME
def pocket_search(structure: str, outputdir: str, pdbdir: str, targetdir: str) -> None:
    """
    The main pocketSearch flow control function. Takes singular elements of the
    total pocketSearch geometric screen and generates surface files, obtains the
    intersection surface with the target and scores all the intersections.
    Inputs:
        structure
        outputdir
        pdbdir
        targetdir
    Outputs:
        None
    """

    print(f'-----Running on: {structure}-----')
    
    # get coordinates of current pocket
    get_homology_diagram()
    obtain_wasserstein()

    # get scores and update scorefile
    extract_score(outputdir, structure)
    
    # append scorefile
    append_scorefile(outputdir, pdbdir, structure, min_intersect, vol)

    # prep for ROSETTA
    rosetta_prep(outputdir, pdbdir, min_intersect, min_hits, structure)

    # move scored structures
    move_scored_structures(outputdir, pdbdir)
    
    # update checkpoint file
    update_checkpoint(outputdir, structure)


#FIXME
def postprocessing(outdir: str) -> None:
    """
    Cleans up scorefile so that it is easier to read.
    Inputs:
        outdir - output directory where scorefile is located
    Outputs:
        None
    """
    score = f'{outdir}score.txt'
    sfile = open(f'{outdir}score.txt', 'r')
    header = sfile.readline()
    contents = [line.split(';') for line in sfile.readlines()]
    sfile.close()

    shutil.move(score, f'{outdir}score.BAK')

    with open(score, 'w') as outfile:
        outfile.write(header)
        for line in contents:
            pdb, pock, tv, pv, iv, ip, nh, method, cof, name = line
            oline = f'{pdb:<6}{pock:<9}{tv:<11.8}{pv:<11.8}{iv:<9.7}{ip:<7.5}'
            oline += f'{nh:>6.4}  {method:<38}{cof}  {name:>17}'

            outfile.write(oline)

    outfile.close()
    os.remove(f'{outdir}score.BAK')

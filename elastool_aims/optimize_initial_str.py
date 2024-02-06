"""
  Elastool -- Elastic toolkit for zero and finite-temperature elastic constants and mechanical properties calculations

  Copyright (C) 2019-2024 by Zhong-Li Liu and Chinedu Ekuma

  This program is free software; you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software Foundation
  version 3 of the License.

  This program is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
  PARTICULAR PURPOSE.  See the GNU General Public License for more details.

  E-mail: zl.liu@163.com, cekuma1@gmail.com

"""

from os import mkdir, chdir
from os.path import isdir
#from ase.io import vasp
from ase.calculators.aims import Aims

from elastool_aims.aims_utils import aims_run, d2k
from elastool_aims.read_input import indict

def optimize_initial_str(pos_conv, cwd, parameters):
    if not isdir('OPT'):
        mkdir('OPT')
    chdir('OPT')

    #vasp.write_vasp('POSCAR', pos_conv, vasp5=True, direct=True)
    pos_conv.write("geometry.in", format="aims", scaled=True)
    #kpoints_file_name = 'KPOINTS-static'

    #calculator = Aims(directory="./", parameters=parameters)
    calculator = Aims("",directory="./")
    calculator.template.write_input(calculator.profile, calculator.directory, pos_conv, parameters, ["energy"])
    #pos_optimized = aims_run("srun /u/yiy/scratch/aims/FHIaims_cbmvbm_cube/build_Feb2_2024_cpu/aims.x > aims.out")
    pos_optimized = aims_run(" ".join(indict['parallel_submit_command']))

    chdir('..')

    return pos_optimized
    

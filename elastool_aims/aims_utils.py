from pathlib import Path

import numpy as np
import os
import time

import spglib as spg

from ase.atoms import Atoms
import json
from ase.io.aims import read_aims
from ase.constraints import FixScaledParametricRelations

from warnings import warn

from ase.calculators.aims import Aims
import ase.io
import ase.io.aims
from monty.json import MontyEncoder

import subprocess


def d2k(atoms, kptdensity=3.5, even=True):
    """Convert k-point density to Monkhorst-Pack grid size.

    inspired by [ase.calculators.calculator.kptdensity2monkhorstpack]

    Parameters
    ----------
    atoms: Atoms object
        Contains unit cell and information about boundary conditions.
    kptdensity: float or list of floats
        Required k-point density.  Default value is 3.5 point per Ang^-1.
    even: bool
        Round up to even numbers.

    Returns
    -------
    list
        Monkhorst-Pack grid size in all directions
    """
    recipcell = atoms.cell.reciprocal()
    return d2k_recipcell(recipcell, atoms.pbc, kptdensity, even)


def d2k_recipcell(recipcell, pbc, kptdensity=3.5, even=True):
    """Convert k-point density to Monkhorst-Pack grid size.

    Parameters
    ----------
    recipcell: ASE Cell object
        The reciprocal cell
    pbc: list of Bools
        If element of pbc is True then system is periodic in that direction
    kptdensity: float or list of floats
        Required k-point density.  Default value is 3.5 point per Ang^-1.
    even: bool
        Round up to even numbers.

    Returns
    -------
    list
        Monkhorst-Pack grid size in all directions
    """
    if not isinstance(kptdensity, list) and not isinstance(kptdensity, np.ndarray):
        kptdensity = 3 * [float(kptdensity)]
    kpts = []
    for i in range(3):
        if pbc[i]:
            k = 2 * np.pi * np.sqrt((recipcell[i] ** 2).sum()) * float(kptdensity[i])
            if even:
                kpts.append(2 * int(np.ceil(k / 2)))
            else:
                kpts.append(int(np.ceil(k)))
        else:
            kpts.append(1)
    return kpts


def to_spglib_cell(atoms):
    """Convert ase.atoms.Atoms to spglib cell

    Args:
      atoms(ase.atoms.Atoms): Atoms to convert

    Returns:
        tuple

    """
    lattice = atoms.cell
    positions = atoms.get_scaled_positions()
    number = atoms.get_atomic_numbers()
    return (lattice, positions, number)


def get_spacegroup(atoms, symprec=1e-5):
    """return spglib spacegroup

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        The structure to get the dataset of
    symprec: float
        The tolerance for determining symmetry and the space group

    Returns
    -------
    str:
        The spglib space group
    """

    return spg.get_spacegroup(to_spglib_cell(atoms), symprec=symprec)


def run_aflow_wrapper(proto, params, outfile, remove_angles=True):
    parameters = [float(param) for param in params.split(",")]

    command = f"aflow --proto={proto} --params={params} --aims --add_equations"
    if remove_angles:
        warn(
            "Adding commands to remove lattice angles from the constraints. This is done using sed so may break in the future if AFLOW corrects the issue themselves.",
            Warning,
        )
        command += " | sed 's/a b c alpha beta gamma/ax bx by cx cy cz/g'"
        command += " | sed 's/a b c beta/ax by cx cz/g'"
        command += " | sed 's/a , 0 , 0/ax , 0 , 0/g'"
        command += " | sed 's/0 , b , 0/0 , by , 0/g'"
        command += " | sed 's/0.5\\*a , -0.5\\*b , 0/0.5\\*ax , -0.5\\*by , 0/g'"
        command += " | sed 's/0.5\\*a , 0.5\\*b , 0/0.5\\*ax , 0.5\\*by , 0/g'"
        command += " | sed 's/*cos(beta)/x/g'"
        command += " | sed 's/*sin(beta)/z/g'"
        command += " | sed 's/*cos(gamma)/x/g'"
        command += " | sed 's/*sin(gamma)/y/g'"

    command += f" > {outfile}"
    os.system(command)

    atoms = read_aims(outfile, apply_constraints=False)
    info_str = open(outfile, "r").readline()[2:]
    info_str += "# With constraints mapped into the unit cell using the aflow_structure_wrapper utility in FHI-aims"

    pos_constraint = None
    for constraint in atoms.constraints:
        if isinstance(constraint, FixScaledParametricRelations):
            pos_constraint = constraint
            break

    if pos_constraint is not None:
        Jacobian = pos_constraint.Jacobian.copy()
        B = pos_constraint.const_shift.copy()
        params = pos_constraint.params.copy()
        if len(params) > 0:
            scaled_pos = ((Jacobian @ parameters[-1 * len(params) :]) + B).reshape(
                (-1, 3)
            )
        else:
            scaled_pos = B.reshape(-1, 3)
        atoms.set_scaled_positions(scaled_pos)

    atoms.set_positions(atoms.get_positions())
    atoms.write(
        outfile, format="aims", geo_constrain=True, scaled=True, info_str=info_str
    )
    return atoms

def save_aims_parameters_json(input_dict, filename):
    json_input = json.dumps(input_dict, indent=2, cls=MontyEncoder)
    with open(filename, 'w') as outfile:
        outfile.write(json_input)
    return

def load_aims_parameters_json(filename):
    parameters_read = {}
    with open(filename) as infile:
        parameters_read = json.load(infile)
    return parameters_read

def is_aims_success():
    f = open("aims.out")
    lines = f.readlines()
    lines = "".join(lines)
    return "Have a nice day." in lines

def aims_run(para_sub_com):
    go = subprocess.Popen(para_sub_com, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while go.poll() is None:
        time.sleep(2)
    assert is_aims_success(), "aims run failed."
    return ase.io.aims.read_aims_output("aims.out")


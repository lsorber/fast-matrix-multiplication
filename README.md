# Fast matrix multiplication

Naive multiplication of two N by N matrices requires N^3 scalar multiplications. For N=2, Strassen showed that it could be done in only R=7 < 8=N^3 multiplications. For N=3, it is known that 19 <= R <= 23, and for N=4 it is known that 34 <= R <= 49. This repository contains code that generates a mixed-integer linear program (MILP) formulation of the fast matrix multiplication problem for finding solutions with R < N^3 and proving that they are optimal. For a more detailed description, check out the accompanying [manuscript](latex/fast-matrix-multiplication.pdf).

## Generating the MILP

To be able to generate MILP instances for various N and R, you need Python 3 along with the packages NumPy and PuLP. You can install these with `conda env create` if you have Anaconda or Miniconda, and then activate the environment with `source activate fast-matrix-multiplication-env`.

Once the environment is activated, simply run `python model.py` to generate a variety of MILP instances and initial solutions in the `instances` directory.

## Solving the MILP

To solve an instance with Gurobi, run `./solve.sh fastxgemm-[parameters].lp` from within the `instances` folder.

# Ballistic deposition

Simulation of off-lattice ballistic deposition.

ballistic_deposition.py is the main Notebook with the interactive figure.

functions.py contains
* a plot function from Stackoverflow for displaying the particles correctly
* the deposition function
* the surface calculator

The notebook containt the widgets and the plot, and then, my attempts at measuring the exponents (since my plots do not look as I expected, this part is still not ready).

Several runs are pickled into the picklestring file, this contains depositions of 100000 particles of size 0.1 onto different lengths at different angles.

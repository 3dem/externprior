# Exploiting prior knowledge about biological macromolecules in cryo-EM structure determination
This repository contains code used in the bioRxiv preprint _Exploiting prior knowledge about biological macromolecules in cryo-EM structure determination_, Dari Kimanius, Gustav Zickert, Takanori Nakane, Jonas Adler, Sebastian Lunz, Carola-Bibiane Schonlieb, Ozan Oktem, Sjors Scheres (doi: https://doi.org/10.1101/2020.03.25.007914).

Code presented here can also be used as an example for the external reconstruction functionality in RELION, which provides a hook for executing alternative reconstruction protocols.

The exrecon_cwred.py should be submitted to the reconstruction process by setting the environmental variable `RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE` to the command that executes the script. Relion will then append the path to the external reconstruction star-file to that command when the executable is called. The star-file contains information about paths to intermediate data files and refinement statistics, e.g. FSC.

Example usage:

`export RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE="python /path/to/exrecon_cwred.py /path/to/model"`
`mpirun -n 3 relion_refine_mpi <flags> --external_reconstruct`

The relion_it module can load and parse star-files for this purpose. An example star-file from a reconstruction is also provided here (`example_external_reconstruct.star`). 
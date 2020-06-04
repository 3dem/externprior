# Exploiting prior knowledge about biological macromolecules in cryo-EM structure determination
This repository contains code used in the bioRxiv preprint _Exploiting prior knowledge about biological macromolecules in cryo-EM structure determination_, Dari Kimanius, Gustav Zickert, Takanori Nakane, Jonas Adler, Sebastian Lunz, Carola-Bibiane Schonlieb, Ozan Oktem, Sjors Scheres (doi: https://doi.org/10.1101/2020.03.25.007914).

Code presented here can also be used as an example for the external reconstruction functionality in RELION, which provides a hook for executing alternative reconstruction protocols.

The exrecon_cwred.py should be submitted to the reconstruction process by setting the environmental variable `RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE` to the command that executes the script. 
Relion will then append the path to the external reconstruction star-file to that command when the executable is called. 
The star-file contains information about paths to intermediate data files and refinement statistics, e.g. FSC.
The relion_it module can load and parse star-files. An example star-file from a reconstruction is also provided in this repository (`example_external_reconstruct.star`).

This repository also contains a pre-trained CNN model for denoising of intermediate RELION reconstructions and an example image stack from the test dataset, PDB: 4BB9, SNR (ii). 

For running a refinement with the pre-trained network, run:
````
wget https://github.com/3dem/externprior/releases/download/example/4bb9_data.tar.gz
wget https://github.com/3dem/externprior/releases/download/example/dncnn_model.tar.gz
tar -xvf 4bb9_data.tar.gz
tar -xvf dncnn_model.tar.gz
git clone https://github.com/3dem/externprior.git
export RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE="python externprior/exrecon_cwred.py dncnn_model"
mkdir cwred_test
mpirun -n 3 relion_refine_mpi --auto_refine --split_random_halves --i 4bb9_data/particles.star --ref 4bb9_data/ref.mrc --ini_high 30 --pad 1 --particle_diameter 110 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym C1 --preread_images --low_resol_join_halves 40 --norm --scale --gpu --j 4 --pool 30 --dont_combine_weights_via_disc --o cwred_test/run --external_reconstruct
````
For this example, you need: RELION 3.1 pre-installed, Tensorflow 1.15 and the Mrcfile python module. 
To run the same reconstruction with regular MAP remove `--external_reconstruct`.
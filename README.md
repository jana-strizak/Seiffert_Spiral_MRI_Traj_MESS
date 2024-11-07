Creating the Seiffert Spiral MRI Trajectroy, with gradient waveforms ready to be input into the MRI.

Create trajectory based on your desired specifications. 
The gradient waveforms, consisting of the gradient x,y,z values at each sampling time dictated by your the inputted sampling rate.

To install: 
1. you will first need to make a virtual environment, and install numpy and Cython==0.29.36.
2. Then you can pip install the package SeiffertSpiralMRITraj by running "pip install ."
3. I use spyder to run python and jupyter notebook, but your interpreter is up to you. 

** inorder to run simulations with MRI phantom, you will need to install BART (Berkley Advanced Reconstruction Toolbox) 
Follow instructions here: https://mrirecon.github.io/bart/installation.html
Make sure BART is added to your commandline so 'BART' is recondnized 

See Examples folder for usable scripts 'makeTrajectory_example.py or .ipynb'


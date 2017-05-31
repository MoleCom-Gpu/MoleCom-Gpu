# Molecular Communication Gpu Simulator (a.k.a. Parallel simulation framework for nanonetworking)

 This project is being developed as a CmpE491/492 senior project at Bogazici University. The aim of the project is to develop a powerful parallel simulation framework utilizing the parallel computing power of Graphical Processing Units (GPU). The project is implemented using Torch scientific framework and its CUDA package CuTorch.
 For more details, please refer to documentation of the project. Project presentations and reports are also in the documentation.
## Specifications and Requirements
 In order to run the simulator on a local machine the following are **required**
 * Ubuntu 12+
 * An NVdia GPU supporting CUDA
 * CUDA (please refer to http://docs.nvidia.com/cuda/cuda-installation-guide-linux/ to install CUDA)
 * Torch Framework (please refer to http://torch.ch/docs/getting-started.html to install Torch)
 
## Use Scenarios
### Configuration
 To configure parameters of the simulation open the file **configGenerate.lua** and edit the parameters. To generate **config** file, run
 
 ```
 th configGenerate.lua
 ```
### Running the Simulator
 After configuring the simulation parameters simply run the script **bm-simulator.lua** by 
 
 ```
 th bm-simulator.lua -o outputfile
 ```
 
 The results will be written into the file **outputfile**. If no filename is specified, the default file name is **result.txt**
 
 The resulting file contains the number of molecules received by each receiver per symbol duration. There are some sample Matlab scripts in scripts folder those using the raw result data for interpretation.
 

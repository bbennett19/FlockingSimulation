# FlockingSimulation
Tested with GCC version 5.3 and CUDA version 8.0  
Term project for CIS 510 Into to Parallel Computing

## How to run
Note on command line arguments (other than the fact that they should probably be redesigned): No arguments are required,   
but you must give arguments up to the argument you want to change.  
Default values are:   
count=100, run time=99999, draw=1, radius=1, separation weight=1, cohesion weight=1, alignment weight=1, destination weight=0.01   
E.g. you want to change the separation weight  
Arguments will look like:   
100(count) 10(run time) 1(draw flag) 1(radius) 1.5(separation weight)  
If running and want to quit press the 'Escape' key, you will still get FPS data.   
If you close the window you will NOT get FPS data  


Serial:  
Command line arguments in order: boid count, run time, draw flag(0 = do not draw), nearest neighbor radius, separation weight, cohesion weight, alignment weight, destination weight  
To run: ./flocking_serial


OpenMP:  
Command line arguments: same as serial  
To run: ./flocking_omp


CUDA:  
Command line arguments: same as serial + DeviceIndex   
On the testing machine I used: DeviceIndex of 0 = Tesla K20c, 1 = Tesla c2075, 2 = GTX 480  
Default DeviceIndex is 0, if you only have one GPU installed this value should never be changed  


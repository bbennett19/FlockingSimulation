all: flocking_serial flocking_omp flocking_cuda
flocking_serial:
	g++ -std=c++11 serial/main.cpp common/Vector2.cpp common/WindowManager.cpp -o flocking_serial -lX11 -lGL -lGLU

flocking_omp:
	g++ -std=c++11 parallel/main.cpp common/Vector2.cpp common/WindowManager.cpp -o flocking_omp -lX11 -lGL -lGLU -fopenmp

flocking_cuda:
	nvcc -std=c++11 --relocatable-device-code true gpu/main.cu gpu/Vector2.cu common/WindowManager.cpp -o flocking_cuda -lX11 -lGL -lGLU -lcudadevrt

clean:
	rm flocking_serial flocking_omp flocking_cuda

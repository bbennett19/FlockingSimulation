// Ben Bennett
// CIS 531
// CUDA version of the flocking algorithm
#include <iostream>
#include <math.h>
#include <chrono>
#include <thread>
#include <string.h>
#include<X11/Xlib.h>
#include<X11/XKBlib.h>
#include<GL/glx.h>
#include<GL/glu.h>
#include "Vector2.cuh"
#include "../common/WindowManager.h"

XWindowAttributes wa;

// Number of boids (optional command line arg)
int count = 100;
// Half the width of the world (-width to width, with 0,0 in the center)
int width = 30;
// Speed of the boids
GLfloat speed = 3;
// Local radius for each boid (optional command line arg)
GLfloat radius = 1.0;
// Weights for separation, cohesion, alignment, and destination (optional command line args)
float sepWeight = 1.0;
float cohWeight = 1.0;
float aliWeight = 1.0;
float destWeight = 0.01; // This weight must be small to work well
// Running time of the simulation (optional command line arg)
float runtime = 99999.0;
// Draw flag (optional command line arg)
bool draw = true;
int deviceIndex = 0;
Vector2 center = Vector2(0.0,0.0);

// Position and direction arrays
Vector2* position;
Vector2* direction;
Vector2* tempPos;
Vector2* tempDir;
Vector2* cu_pos;
Vector2* cu_dir;
Vector2* cu_tempPos;
Vector2* cu_tempDir;


// Copy array utility function
void copy(Vector2 from[], Vector2 to[], int length) {
    for(int i = 0; i < length; i++) {
        to[i] = from[i];
    }
}

// Draw a red triangle
void drawTriangle(GLfloat size) {
    glBegin(GL_TRIANGLES);

    glColor3f(204./255.,0.0,0.0);
    glVertex3f(-size/2.,0.0,0.0);
    glVertex3f(size/2.,0.0,0.0);
    glVertex3f(0.0,size,0.0);

    glEnd();
}

// Draw all the boids. This could be improved using instancing
void drawBoids(WindowManager* mgr) {
    float  aspect_ratio;

    XGetWindowAttributes(mgr->getDisplay(), mgr->getWindow(), &wa);
    glViewport(0, 0, wa.width, wa.height);
    aspect_ratio = (float)(wa.width) / (float)(wa.height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-width*aspect_ratio, width*aspect_ratio, -width, width, 1., 100.);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0., 0., -10, 0., 0., 0., 0., 1., 0.);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    for(int i = 0; i < count; i++) {
        glPushMatrix();
        glTranslatef(position[i].x, position[i].y, 0.0);
        drawTriangle(0.20f);
        glPopMatrix();
    }
    glXSwapBuffers(mgr->getDisplay(), mgr->getWindow());
}

// Process the command line args
void processArgs(int argc, char* argv[]) {
    if(argc > 1)
        count = atoi(argv[1]);
    if(argc > 2)
        runtime = std::stof(argv[2]);
    if(argc > 3)
        draw = (bool)atoi(argv[3]);
    if(argc > 4)
        radius = std::stof(argv[4]);
    if(argc > 5)
        sepWeight = std::stof(argv[5]);
    if(argc > 6)
        cohWeight = std::stof(argv[6]);
    if(argc > 7)
        aliWeight = std::stof(argv[7]);
    if(argc > 8)
        destWeight = std::stof(argv[8]);
    if(argc > 9)
        deviceIndex = atoi(argv[9]);
}

// Initialize the position and direction of each boid randomly
void init() {
    srand(time(NULL));
    position = new Vector2[count];
    direction = new Vector2[count];
    tempPos = new Vector2[count];
    tempDir = new Vector2[count];

    // Randomize starting position and direction
    for(int i=0; i < count; i++) {
        position[i] = Vector2((GLfloat)rand()/(GLfloat)RAND_MAX*2.0*width - width, (GLfloat)rand()/(GLfloat)RAND_MAX*2.0*width - width);
        GLfloat angle = (GLfloat)rand()/(GLfloat)RAND_MAX * M_PI*2.0;
        direction[i] = Vector2(cos(angle), sin(angle));
        tempPos[i] = position[i];
        tempDir[i] = direction[i];
    }
}

__device__
// Calculate the separation, cohesion, and alignment vectors
void calculateProperties(Vector2& sep, Vector2& coh, Vector2& ali, int index, int c, float radius, Vector2* cu_pos, Vector2* cu_dir) {
    int num = 0;
    for(int i=0; i < c; i++) {
        if(i != index) {
            Vector2 diff = cu_pos[index]-cu_pos[i];
            if(diff.magnitudeSquared() < radius*radius) {
                float d = diff.magnitude();
                num++;
                ali = ali + cu_dir[i];
                coh = coh + cu_pos[i];
                sep = sep + ((diff)*(1.0/d));
            }
        }
    }
    if(num > 0) {
        float z = 1.0/(float)num;
        ali = ali*z;
        coh = (coh*z)-cu_pos[index];
        sep = sep*z;
    }
}

// CUDA kernel. Updates the position and direction for a single boid. Updates based on the rules for
// cohesion, separation, and alignment, as well as a fourth rule to keep in screen
__global__
void updateBoids(float deltaTime, float speed, float sepWeight, float cohWeight, float aliWeight, float destWeight,
                 float radius, int c, Vector2* cu_pos, Vector2* cu_dir, Vector2* cu_tempPos, Vector2* cu_tempDir) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c) {
        Vector2 sep;
        Vector2 coh;
        Vector2 ali;
        calculateProperties(sep, coh, ali, i, c, radius, cu_pos, cu_dir);

        Vector2 dest = Vector2(0.0,0.0)-cu_pos[i];
        Vector2 dir = (((sep * sepWeight) + (coh * cohWeight) + (ali * aliWeight) + (dest * destWeight)) * .25);

        if (dir.x != 0.0 && dir.y != 0.0)
            cu_tempDir[i] = dir.normalize();
        cu_tempPos[i] = cu_tempPos[i] + cu_tempDir[i] * speed * deltaTime;
    }
}

int main(int argc, char* argv[]) {
    processArgs(argc, argv);
    std::cout << "Count:" << count << " RunTime:" << runtime << " Draw:" << draw << " Radius:" << radius <<
              " SeparationWeight:" << sepWeight << " CohesionWeight:" << cohWeight << " AlignmentWeight:" << aliWeight
              << " DestinationWeight:" << destWeight << " DeviceIndex: " << deviceIndex << std::endl;
    WindowManager mgr;
    mgr.createWindow();
    cudaSetDevice(deviceIndex);
    cudaDeviceProp props;
    int device = 0;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    std::cout << "Using device " << device << ": " << props.name << std::endl;

    // Boid initialization
    init();

    // Allocate memory on the GPU
    cudaMalloc(&cu_dir, sizeof(Vector2)*count);
    cudaMalloc(&cu_pos, sizeof(Vector2)*count);
    cudaMalloc(&cu_tempDir, sizeof(Vector2)*count);
    cudaMalloc(&cu_tempPos, sizeof(Vector2)*count);

    // Copy the position and direction values to the GPU
    cudaMemcpy(cu_tempDir, tempDir, sizeof(Vector2)*count, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_tempPos, tempPos, sizeof(Vector2)*count, cudaMemcpyHostToDevice);

    // Setup timer
    float deltaTime = 0.0;
    float elapsedTime = 0.0;
    int frameCount = 0;
    std::chrono::high_resolution_clock::time_point start, end;

    // Run until the user quits or the designated runtime is reached
    while(elapsedTime < runtime && !mgr.quit()) {
        start = std::chrono::high_resolution_clock::now();
        // Copy the latest position and direction data to the GPU
        cudaMemcpy(cu_dir, direction, sizeof(Vector2)*count, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_pos, position, sizeof(Vector2)*count, cudaMemcpyHostToDevice);

        // Update the position and direction on the GPU
        updateBoids<<<(count+255)/256, 256>>>(deltaTime, speed, sepWeight, cohWeight, aliWeight, destWeight,
                radius, count, cu_pos, cu_dir, cu_tempPos, cu_tempDir);

        // Get the updated position and direction for drawing
        cudaMemcpy(position, cu_tempPos, sizeof(Vector2)*count, cudaMemcpyDeviceToHost);
        cudaMemcpy(direction, cu_tempDir, sizeof(Vector2)*count, cudaMemcpyDeviceToHost);

        if(draw)
            drawBoids(&mgr);
        end = std::chrono::high_resolution_clock::now();
        elapsedTime += deltaTime;
        frameCount++;
        deltaTime = std::chrono::duration_cast<std::chrono::duration<float>>(end-start).count();
    }

    // Print the frames per second
    std::cout << "Average FPS: " << frameCount/elapsedTime << std::endl;
    cudaFree(cu_dir);
    cudaFree(cu_pos);
    cudaFree(cu_tempDir);
    cudaFree(cu_tempPos);
    mgr.shutdown();
    return 0;
}
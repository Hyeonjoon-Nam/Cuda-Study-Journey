#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include "kernel.cuh" 

// OpenGL Handles
GLuint vbo; // Vertex Buffer Object
GLuint vao; // Vertex Array Object

// CUDA Graphics Resource Handle
struct cudaGraphicsResource* cuda_vbo_resource;

// Simulation Parameters
// Using 64x64 (4096) particles for Naive O(N^2) implementation.
const int mesh_width = 64;
const int mesh_height = 64;
const int num_particles = mesh_width * mesh_height;

float anim_time = 0.0f;

// ------------------------------------------------------------------
// Initialize OpenGL Buffers & Register with CUDA
// ------------------------------------------------------------------
void initGL() {
    // 1. Create and bind VAO (Essential for Core Profile)
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // 2. Create and bind VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // 3. Allocate memory on GPU (GL_DYNAMIC_DRAW for frequent updates)
    unsigned int size = num_particles * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    // Unbind buffer (VAO keeps the state)
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // 4. Register VBO with CUDA
    initCuda(&cuda_vbo_resource, vbo, num_particles);
}

// ------------------------------------------------------------------
// Main Rendering Loop
// ------------------------------------------------------------------
void display() {
    anim_time += 0.01f;

    // Step 1: Run CUDA Kernel to update particle positions
    runCuda(cuda_vbo_resource, num_particles, anim_time);

    // Step 2: Render particles using OpenGL
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black background

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
    // Define layout: 4 floats per vertex (x, y, z, w)
    glVertexPointer(4, GL_FLOAT, 0, (void*)0);
    glEnableClientState(GL_VERTEX_ARRAY);

    // Draw settings
    glPointSize(3.0f);            // Make points visible
    glColor3f(0.0f, 1.0f, 1.0f);  // Cyan color
    glDrawArrays(GL_POINTS, 0, num_particles);

    // Cleanup state
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Swap buffers (Double Buffering)
    glutSwapBuffers();
    
    // Request next frame
    glutPostRedisplay();
}

// ------------------------------------------------------------------
// Main Entry Point
// ------------------------------------------------------------------
int main(int argc, char** argv) {
    // 1. Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(1024, 768);
    glutCreateWindow("Project 06: CUDA Interop Boids (Naive O(N^2))");

    // 2. Initialize GLEW
    glewInit();

    // 3. Initialize Graphics & Compute
    initGL();

    // 4. Register Callback & Start Loop
    glutDisplayFunc(display);
    glutMainLoop();

    // 5. Cleanup
    cleanupCuda(cuda_vbo_resource);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);

    return 0;
}
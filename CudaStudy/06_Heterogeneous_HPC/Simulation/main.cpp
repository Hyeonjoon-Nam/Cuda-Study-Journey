#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include "kernel.cuh" 

GLuint vbo; // Vertex Buffer Object (OpenGL Memory Handle)
struct cudaGraphicsResource* cuda_vbo_resource; // CUDA Handle for the same memory

const int mesh_width = 1024;
const int mesh_height = 1024;
const int num_particles = mesh_width * mesh_height;

float anim_time = 0.0f;

void initGL() {
    // 1. Create a VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // 2. Allocate memory on GPU (but don't upload CPU data, passing NULL)
    unsigned int size = num_particles * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // 3. Register this VBO to CUDA
    initCuda(&cuda_vbo_resource, vbo, num_particles);
}

void display() {
    anim_time += 0.01f;

    // [Interop Core] Let CUDA update the buffer content
    runCuda(cuda_vbo_resource, num_particles, anim_time);

    // Clear Screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // [Render] OpenGL renders directly from the VBO updated by CUDA
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glColor3f(0.0f, 1.0f, 1.0f); // Cyan Color
    glDrawArrays(GL_POINTS, 0, num_particles);

    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glutSwapBuffers();
    glutPostRedisplay();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(1024, 768);
    glutCreateWindow("Project 06: CUDA Interop Wave (1 Million Particles)");

    glewInit();
    initGL();

    glutDisplayFunc(display);
    glutMainLoop();

    cleanupCuda(cuda_vbo_resource);
    glDeleteBuffers(1, &vbo);
    return 0;
}
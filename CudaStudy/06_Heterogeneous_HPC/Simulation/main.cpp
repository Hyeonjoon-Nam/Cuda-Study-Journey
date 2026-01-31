#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include <string> // Title update용
#include "kernel.cuh" 

// OpenGL Handles
GLuint vbo;
GLuint vao;

// CUDA Resource
struct cudaGraphicsResource* cuda_vbo_resource;

// ------------------------------------------------------------------
// [CHANGE] Massive Scale Up!
// ------------------------------------------------------------------
// From 64x64 (4,096) -> 512x512 (262,144 particles)
// RTX 3070 should handle this easily with Uniform Grid.
const int mesh_width = 512;
const int mesh_height = 512;
const int num_particles = mesh_width * mesh_height;

float anim_time = 0.0f;
int fps_frame_count = 0;
int fps_time_base = 0;

void initGL() {
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    unsigned int size = num_particles * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    initCuda(&cuda_vbo_resource, vbo, num_particles);
}

void calculateFPS() {
    fps_frame_count++;
    int current_time = glutGet(GLUT_ELAPSED_TIME);

    if (current_time - fps_time_base > 1000) {
        float fps = fps_frame_count * 1000.0f / (current_time - fps_time_base);
        std::string title = "Project 06: Uniform Grid Boids | Particles: " 
                            + std::to_string(num_particles) 
                            + " | FPS: " + std::to_string((int)fps);
        glutSetWindowTitle(title.c_str());
        
        fps_time_base = current_time;
        fps_frame_count = 0;
    }
}

void display() {
    anim_time += 0.01f;

    // 1. CUDA Physics Update
    runCuda(cuda_vbo_resource, num_particles, anim_time);

    // 2. Render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, (void*)0);
    glEnableClientState(GL_VERTEX_ARRAY);

    // [CHANGE] Reduce point size because there are too many particles
    glPointSize(1.0f); 
    glColor3f(0.0f, 1.0f, 1.0f); // Cyan
    glDrawArrays(GL_POINTS, 0, num_particles);

    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glutSwapBuffers();
    glutPostRedisplay();
    
    // 3. FPS Calculation
    calculateFPS();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(1024, 768);
    glutCreateWindow("Project 06: CUDA Boids"); // Title will be updated by calculateFPS

    glewInit();
    initGL();

    glutDisplayFunc(display);
    glutMainLoop();

    cleanupCuda(cuda_vbo_resource);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);

    return 0;
}
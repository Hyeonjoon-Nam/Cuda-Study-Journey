#include <GL/glew.h>
#include <GL/freeglut.h>
#include <string> // Title update
#include "kernel.cuh" 
#include <thread>
#include <atomic>
#include "SerialPort.h"

// OpenGL Handles
GLuint vbo;

// CUDA Resource
struct cudaGraphicsResource* cuda_vbo_resource;

// 
std::atomic<int> g_sensorValue = 0;

// Simulation Settings
// 128x128 = 16,384 particles (Best for visualization)
const int mesh_width = 128;
const int mesh_height = 128;
const int num_particles = mesh_width * mesh_height;

float anim_time = 0.0f;
int fps_frame_count = 0;
int fps_time_base = 0;

void initGL() {
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

// Thread Worker for Serial Communication
void serial_worker() {
    const char* portname = "\\\\.\\COM3";
    SerialPort arduino(portname);
    char buffer[256];

    while (true) {

        if (arduino.isConnected()) {
            if (arduino.readSerialPort(buffer, sizeof(buffer))) {
                g_sensorValue = atoi(buffer);
                // Optional: Print for debugging
                printf("\nsensor value: %d", g_sensorValue.load());
            }
            Sleep(10); // Prevent CPU hogging
        }
    }
}

void display() {
    anim_time += 0.01f;
    
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

    // 1. CUDA Physics Update
    int current_sensor = g_sensorValue.load();
    runCuda(cuda_vbo_resource, num_particles, anim_time, current_sensor);

    // 2. Render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, (void*)0);
    glEnableClientState(GL_VERTEX_ARRAY);

    // Set point size
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

    // Enable loop return on close
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

    glewInit();
    initGL();

    glutDisplayFunc(display);

    // Launch Serial Thread
    std::thread receiver(serial_worker);
    receiver.detach();

    glutMainLoop();

    // Cleanup
    cleanupCuda(cuda_vbo_resource);
    glDeleteBuffers(1, &vbo);

    return 0;
}
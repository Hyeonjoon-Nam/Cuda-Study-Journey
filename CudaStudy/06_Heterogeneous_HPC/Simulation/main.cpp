#include <GL/glew.h>
#include <GL/freeglut.h>
#include <string> 
#include "kernel.cuh" 
#include <thread>
#include <atomic>
#include "SerialPort.h"
#include <iostream>

// OpenGL Handles
GLuint vbo;

// CUDA Resource for OpenGL interoperability
struct cudaGraphicsResource* cuda_vbo_resource;

// Atomic variable ensures thread-safe data sharing between the Serial Thread and the Render Thread.
// Prevents race conditions when reading/writing the sensor value simultaneously.
std::atomic<int> g_sensorValue = 0;

// Simulation Settings
// 128x128 = 16,384 particles (Optimal for visualization and performance)
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

    // Register the OpenGL VBO with CUDA
    initCuda(&cuda_vbo_resource, vbo, num_particles);
}

void calculateFPS() {
    fps_frame_count++;
    int current_time = glutGet(GLUT_ELAPSED_TIME);

    if (current_time - fps_time_base > 1000) {
        float fps = fps_frame_count * 1000.0f / (current_time - fps_time_base);
        std::string title = "Project 06: Heterogeneous Boids | Particles: "
            + std::to_string(num_particles)
            + " | FPS: " + std::to_string((int)fps);
        glutSetWindowTitle(title.c_str());

        fps_time_base = current_time;
        fps_frame_count = 0;
    }
}

// Thread Worker for Async Serial Communication
void serial_worker() {
    const char* portname = "\\\\.\\COM3";
    SerialPort arduino(portname);

    if (!arduino.isConnected()) {
        std::cerr << "ERROR: Failed to connect arduino. Check the port number." << std::endl;
        return;
    }

    std::cout << "Arduino Connected via SerialPort Class!" << std::endl;

    char buffer[256];
    // Static string retains partial data chunks across multiple read cycles
    static std::string receivedString = "";

    while (arduino.isConnected()) {
        // Read incoming serial data chunk
        int bytesRead = arduino.readSerialPort(buffer, sizeof(buffer));

        if (bytesRead > 0) {
            // Process the chunk character by character
            for (int i = 0; i < bytesRead; i++) {
                char c = buffer[i];

                if (c == '\n') { // End of a complete message packet
                    size_t sPos = receivedString.find("S:");
                    if (sPos != std::string::npos) {
                        try {
                            // Extract the numeric string and convert to integer
                            std::string numPart = receivedString.substr(sPos + 2);
                            int val = std::stoi(numPart);

                            // Safely update the global sensor value
                            g_sensorValue.store(val);

                            // std::cout << "Sensor: " << val << std::endl; // Debug
                        }
                        catch (...) {
                            // Catch format exceptions caused by corrupted serial data
                        }
                    }
                    // Clear the buffer to prepare for the next message
                    receivedString = "";
                }
                else {
                    // Append characters to build the complete message
                    receivedString += c;
                }
            }
        }

        // Slight delay to prevent the worker thread from monopolizing the CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void display() {
    anim_time += 0.01f;

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

    // 1. Fetch the latest sensor value from the atomic variable safely
    int current_sensor = g_sensorValue.load();

    // 2. CUDA Physics Update (Pass hardware input to GPU)
    runCuda(cuda_vbo_resource, num_particles, anim_time, current_sensor);

    // 3. Render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, (void*)0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glPointSize(1.0f);
    glColor3f(0.0f, 1.0f, 1.0f); // Cyan
    glDrawArrays(GL_POINTS, 0, num_particles);

    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glutSwapBuffers();
    glutPostRedisplay();

    // 4. Update Window Title with FPS
    calculateFPS();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(1024, 768);
    glutCreateWindow("Project 06: CUDA Boids");

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

    glewInit();
    initGL();

    glutDisplayFunc(display);

    // Launch Serial Thread independently so it doesn't block the OpenGL render loop
    std::thread receiver(serial_worker);
    receiver.detach();

    glutMainLoop();

    // Cleanup resources upon exit
    cleanupCuda(cuda_vbo_resource);
    glDeleteBuffers(1, &vbo);

    return 0;
}
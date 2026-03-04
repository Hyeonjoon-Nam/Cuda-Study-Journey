#include <GL/glew.h>
#include <GL/freeglut.h>
#include <string> 
#include <thread>
#include <atomic>
#include <iostream>
#include <queue>
#include <vector>

#include "kernel.cuh" 
#include "SerialPort.h" // Retaining for legacy support
#include "UdpReceiver.h"

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

// Lightweight single-header library used to parse the floor plan image into a pixel array.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Host-side buffer for the warehouse layout (0=navigable, 1=obstacle).
// Retained in memory to visualize the static environment via OpenGL.
unsigned char g_cpu_map[128 * 128];
bool g_map_loaded = false;

// Host-side distance field generated via BFS and target coordinates.
unsigned short g_cpu_dist_map[128 * 128];
int goal_x = -1;
int goal_y = -1;

void initGL() {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    unsigned int size = num_particles * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register the OpenGL VBO with CUDA
    initCuda(&cuda_vbo_resource, vbo, num_particles);
}

// Parses the warehouse floor plan image, thresholds grayscale values to binary states,
// and dispatches the structural data to the GPU constant memory.
void loadMapFromFile() {
    int width, height, channels;
    unsigned char* img = stbi_load("assets/map.png", &width, &height, &channels, 3);

    if (img == NULL) {
        std::cerr << "Warning: map.png not found. Using empty map." << std::endl;
        memset(g_cpu_map, 0, sizeof(g_cpu_map));
    }
    else {
        std::cout << "map.png loaded successfully (" << width << "x" << height << ")" << std::endl;

        // Parse RGB pixels: Red(Target), Black(Wall), White(Path).
        for (int x = 0; x < 128; ++x) {
            for (int y = 0; y < 128; ++y) {
                int img_idx = (y * width + x) * 3;

                int r = img[img_idx];
                int g = img[img_idx + 1];
                int b = img[img_idx + 2];

                // If the cell is nearly 'Red', set it to be the goal.
                if (r > 200 && g < 50 && b < 50) {
                    goal_x = x;
                    goal_y = y;
                    g_cpu_map[y * 128 + x] = 0;
                }
                // If the cell is dark, set it to be a wall.
                else if (r < 128 && g < 128 && b < 128) {
                    g_cpu_map[y * 128 + x] = 1;
                }
                // Else, it's a path.
                else {
                    g_cpu_map[y * 128 + x] = 0;
                }
            }
        }
        stbi_image_free(img);
        g_map_loaded = true;
    }
}

// Generates a discrete distance field using Breadth-First Search (BFS).
// Creates a gradient that flows from all navigable cells down to the target goal(0).
void generateDistanceMap() {
    memset(g_cpu_dist_map, 0xFF, sizeof(g_cpu_dist_map));

    g_cpu_dist_map[goal_y * 128 + goal_x] = 0;

    std::queue<std::pair<int, int>> q;
    q.push({ goal_x, goal_y });
    
    int dx[8] = { 0, 0, -1, 1, -1, 1, -1, 1 };
    int dy[8] = { -1, 1, 0, 0, -1, -1, 1, 1 };

    while (!q.empty()) {
        auto& cur = q.front(); q.pop();
        int x = cur.first;
        int y = cur.second;
        
        int current_dist = g_cpu_dist_map[y * 128 + x];

        for (int i = 0; i < 8; ++i) {
            int nx = x + dx[i];
            int ny = y + dy[i];

            if (nx < 0 || nx >= 128 ||  // Boundary check
                ny < 0 || ny >= 128 ||
                g_cpu_map[ny * 128 + nx] == 1 || // If it's a wall, continue
                g_cpu_dist_map[ny * 128 + nx] <= current_dist + 1 // If existing path is better
                ) continue;

            g_cpu_dist_map[ny * 128 + nx] = current_dist + 1;
            q.push({ nx, ny });
        }

    }

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

void udp_worker() {
    unsigned short port = 8888;

    UdpReceiver esp32(port);

    if (!esp32.isConnected()) {
        std::cerr << "ERROR: Failed to initialize UDP receiver. Check the port number." << std::endl;
        return;
    }

    char buffer[256];
    while (esp32.isConnected()) {
        // Blocks until a wireless packet is received
        int bytesRead = esp32.readUDP(buffer, sizeof(buffer));

        if (bytesRead > 0) {
            try {
                // Parse the numerical string and update shared state
                int val = std::stoi(buffer);
                g_sensorValue.store(val);
            }
            catch (...) {
                // Ignore corrupted or malformed packets to maintain stability
            }
        }
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

    // Render static environmental obstacles and the target goal
    if (g_map_loaded) {
        glPointSize(4.0f); // Scale to fill the 128x128 visual grid
        glBegin(GL_POINTS);
        for (int y = 0; y < 128; ++y) {
            for (int x = 0; x < 128; ++x) {
                int dist = g_cpu_dist_map[y * 128 + x];

                if (dist == 0) {
                    glColor3f(1.0f, 0.f, 0.f);
                }
                else if (g_cpu_map[y * 128 + x] == 1) {
                    glColor3f(1.0f, 1.0f, 1.0f);
                }
                else {
                    glColor3f(0.2f, 0.2f, 0.2f);
                }

                float gl_x = ((x + 0.5f) / 64.0f) - 1.0f;
                float gl_y = ((y + 0.5f) / 64.0f) - 1.0f;
                glVertex2f(gl_x, gl_y);
            }
        }
        glEnd();
    }

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
    loadMapFromFile();
    generateDistanceMap();
    // Dispatch to GPU Constant Memory
    initMapData(g_cpu_map, g_cpu_dist_map);

    glutDisplayFunc(display);

    // Launch Serial Thread independently so it doesn't block the OpenGL render loop
    std::thread receiver(udp_worker); // Change to serial_worker to restore the legacy serial port
    receiver.detach();

    glutMainLoop();

    // Cleanup resources upon exit
    cleanupCuda(cuda_vbo_resource);
    glDeleteBuffers(1, &vbo);

    return 0;
}
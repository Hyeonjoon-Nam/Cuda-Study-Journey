/*
 * 06_MassiveBoids - OpenGL Setup & Test Render
 * Basic OpenGL window creation and primitive rendering test.
 */

#include <iostream>

 // [Important] GLEW must be included before FreeGLUT
#include <GL/glew.h>
#include <GL/freeglut.h>

void display() {
    // 1. Clear the screen (color buffer)
    glClear(GL_COLOR_BUFFER_BIT);

    // 2. Set background color
    glClearColor(0.3f, 0.0f, 0.1f, 1.0f);

    // 3. Draw a single Triangle (representing a Boid)
    // Note: Immediate mode is used here for testing purposes.
    //       Later implementations will use CUDA-OpenGL interop.
    glLoadIdentity();

    glBegin(GL_TRIANGLES);
    glColor3f(0.0f, 1.0f, 1.0f); // Color: Cyan

    // Vertex coordinates (x, y)
    glVertex2f(0.0f, 0.1f);     // Top
    glVertex2f(-0.05f, -0.05f); // Bottom Left
    glVertex2f(0.05f, -0.05f);  // Bottom Right
    glEnd();

    // 4. Swap double buffers (Update screen)
    glutSwapBuffers();
}

int main(int argc, char** argv) {
    // 1. Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(1280, 720);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Project 06: Massive Boids - Setup Complete");

    // 2. Initialize GLEW (Must be done AFTER window creation)
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
        return -1;
    }

    // Output OpenGL version info
    std::cout << ">>> GLEW Version: " << glewGetString(GLEW_VERSION) << std::endl;
    std::cout << ">>> OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

    // 3. Register rendering callback
    glutDisplayFunc(display);

    // 4. Enter main loop
    std::cout << ">>> OpenGL Setup Success. Rendering loop started." << std::endl;
    glutMainLoop();

    return 0;
}
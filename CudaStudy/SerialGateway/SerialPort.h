#pragma once
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// A simple Win32 API wrapper class for serial communication
class SerialPort {
public:
    SerialPort(const char* portName);
    ~SerialPort();

    // Reads data from the serial port into the provided buffer
    // Returns the number of bytes successfully read.
    int readSerialPort(char* buffer, unsigned int buf_size);

    bool isConnected();
    void closeSerial();

private:
    HANDLE hComm; // Win32 Handle to the COM port
    bool connected;
};
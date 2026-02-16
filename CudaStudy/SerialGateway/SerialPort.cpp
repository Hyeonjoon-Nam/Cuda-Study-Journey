#include "SerialPort.h"

SerialPort::SerialPort(const char* portName) {
    // Open the serial port using Win32 CreateFile API
    hComm = CreateFileA(
        portName,
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        0,
        NULL
    );

    if (hComm == INVALID_HANDLE_VALUE) {
        printf("Error in opening serial port\n\n");

        CHAR error_message[256];
        printf("Error");
        DWORD errors = GetLastError();
        FormatMessageA(
            FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            errors,
            0,
            error_message,
            sizeof(error_message),
            NULL
        );
        printf("\nERROR: %s", error_message);
        connected = false;
    }
    else {
        printf("Opening serial port successful\n\n");
        connected = true;

        // Configure COM port parameters using the Device Control Block (DCB)
        DCB DCB_Struct_Parameter = { 0 };
        DCB_Struct_Parameter.DCBlength = sizeof(DCB_Struct_Parameter);
        BOOL status = GetCommState(hComm, &DCB_Struct_Parameter);

        if (status == FALSE)
            printf("\nError in GetCommState()");
        else
            printf("\nGetCommState() Success");

        // Match these settings with the Arduino firmware (9600 baud, 8N1)
        DCB_Struct_Parameter.BaudRate = 9600;
        DCB_Struct_Parameter.ByteSize = 8;
        DCB_Struct_Parameter.Parity = NOPARITY;
        DCB_Struct_Parameter.StopBits = ONESTOPBIT;

        status = SetCommState(hComm, &DCB_Struct_Parameter);
        if (status == FALSE)
            printf("\nError in SetCommState()");
        else
            printf("\nSetCommState() Success");

        // Set communication timeouts to prevent ReadFile from blocking indefinitely
        // This allows the read thread to return quickly if no data is available
        COMMTIMEOUTS timeouts = { 0 };
        timeouts.ReadIntervalTimeout = 20;
        timeouts.ReadTotalTimeoutMultiplier = 1;
        timeouts.ReadTotalTimeoutConstant = 50;
        timeouts.WriteTotalTimeoutMultiplier = 1;
        timeouts.WriteTotalTimeoutConstant = 50;
        SetCommTimeouts(hComm, &timeouts);
    }
}

SerialPort::~SerialPort() {
    if (connected) closeSerial();
}

int SerialPort::readSerialPort(char* buffer, unsigned int buf_size) {
    BOOL success;
    DWORD bytesRead;

    // Read available bytes into the buffer
    // Reads up to (buf_size - 1) to leave room for the null terminator
    success = ReadFile(
        hComm,
        buffer,
        buf_size - 1,
        &bytesRead,
        NULL
    );

    // Null-terminate the string safely
    buffer[bytesRead] = 0;

    return (int)bytesRead;
}

bool SerialPort::isConnected() {
    return connected;
}

void SerialPort::closeSerial() {
    CloseHandle(hComm);
}
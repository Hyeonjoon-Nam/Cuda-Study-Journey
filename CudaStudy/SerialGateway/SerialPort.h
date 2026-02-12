#pragma once
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

class SerialPort {
public:
	SerialPort(const char* portName);
	~SerialPort();

	int readSerialPort(char* buffer, unsigned int buf_size);
	bool isConnected();
	void closeSerial();

private:
	HANDLE hComm;
	bool connected;
};
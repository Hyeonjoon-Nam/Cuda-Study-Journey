#include <Windows.h>
#include <stdio.h>
#include "SerialPort.h"

int main()
{
	const char* portname = "\\\\.\\COM3";
	SerialPort arduino(portname);

	printf("\nListening for Arduino...\n");
	while (true) {
		char buffer[256];
		if (arduino.isConnected()) {
			if (arduino.readSerialPort(buffer, sizeof(buffer)))
			{
				int value = atoi(buffer);
				printf("\nSensor Value: %d", value);
			}
			Sleep(1000);
		}
	}

	return 0;
}
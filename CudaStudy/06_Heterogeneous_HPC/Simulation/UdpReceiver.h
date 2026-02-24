#pragma once
#include <iostream>
#include <winsock2.h>
#include <string>

// Link the WinSock2 library automatically
#pragma comment(lib, "ws2_32.lib")

// Handles wireless data reception via UDP using WinSock2.
class UdpReceiver {
public:
    UdpReceiver(USHORT port);
    ~UdpReceiver();

    bool isConnected() const;
    int readUDP(char* buffer, unsigned int buf_size);

private:
    WSADATA wsaData;
    SOCKET recvSocket;
    sockaddr_in recvAddr;
    bool connected;
};
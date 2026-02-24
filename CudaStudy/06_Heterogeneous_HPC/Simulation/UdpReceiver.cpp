#include "UdpReceiver.h"

UdpReceiver::UdpReceiver(USHORT port) : connected(false), recvSocket(INVALID_SOCKET) {
    // Initialize Windows Sockets API
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WinSock2 Initialization Failed." << std::endl;
        return;
    }

    // Create a Datagram Socket for UDP
    recvSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (recvSocket == INVALID_SOCKET) {
        std::cerr << "Socket Creation Failed." << std::endl;
        WSACleanup();
        return;
    }

    // Configure receiver address and bind to the specified port
    recvAddr.sin_family = AF_INET;
    recvAddr.sin_port = htons(port);
    recvAddr.sin_addr.s_addr = htonl(INADDR_ANY); // Listen on all network interfaces

    if (bind(recvSocket, (SOCKADDR*)&recvAddr, sizeof(recvAddr)) == SOCKET_ERROR) {
        std::cerr << "Socket Binding Failed. Port might be in use." << std::endl;
        closesocket(recvSocket);
        WSACleanup();
        return;
    }

    std::cout << "UDP Receiver active on port " << port << std::endl;
    connected = true;
}

UdpReceiver::~UdpReceiver() {
    if (recvSocket != INVALID_SOCKET) {
        closesocket(recvSocket);
    }
    WSACleanup();
}

bool UdpReceiver::isConnected() const {
    return connected;
}

int UdpReceiver::readUDP(char* buffer, unsigned int buf_size) {
    sockaddr_in senderAddr;
    int senderAddrSize = sizeof(senderAddr);

    // Blocking call: Waits until a UDP datagram arrives
    int bytesReceived = recvfrom(recvSocket, buffer, buf_size - 1, 0, (SOCKADDR*)&senderAddr, &senderAddrSize);

    if (bytesReceived > 0) {
        buffer[bytesReceived] = '\0'; // Ensure the received packet is a valid C-string
    }

    return bytesReceived;
}
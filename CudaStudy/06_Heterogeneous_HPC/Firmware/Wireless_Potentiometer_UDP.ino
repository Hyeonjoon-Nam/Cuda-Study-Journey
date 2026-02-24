#include <WiFi.h>
#include <WiFiUdp.h>

// 1. Wi-Fi (SoftAP) Configuration
const char* ssid = "HPC_Network";       // SSID for the hosted Wi-Fi network
const char* password = "password123";   // Password (minimum 8 characters)

// 2. UDP Communication Settings
WiFiUDP udp;
const char* targetIP = "192.168.4.255"; // Broadcast to all devices in the 192.168.4.x range
const int targetPort = 8888;            // Destination port on the Host PC

// 3. Hardware Configuration
const int POT_PIN = 4; // Analog input pin connected to the potentiometer

void setup() {
    Serial.begin(115200);

    // Initialize the ESP32 as a Software Access Point
    Serial.println("\nStarting SoftAP...");
    WiFi.softAP(ssid, password);

    IPAddress IP = WiFi.softAPIP();
    Serial.print("Access Point IP: ");
    Serial.println(IP); // Typically 192.168.4.1

    // Set ADC resolution to 12-bit (0-4095)
    analogReadResolution(12);
}

void loop() {
    // Read sensor value from hardware
    int potValue = analogRead(POT_PIN);

    // Encapsulate and transmit the data via UDP (Fire-and-forget)
    udp.beginPacket(targetIP, targetPort);
    udp.print(potValue);
    udp.endPacket();

    // Local debugging via Serial Monitor
    Serial.println(potValue);

    // Transmit at approximately 100Hz frequency
    delay(10);
}
#include <avr/io.h>
#include <util/delay.h>

// Baud Rate Calculation (F_CPU is defined by Arduino IDE as 16000000UL)
#define BAUD 9600
// Formula to calculate the UBRR register value for the target baud rate
#define UBRR_VALUE ((F_CPU / 16 / BAUD) - 1)

void setup_uart() {
    // 1. Set Baud Rate (9600 bps)
    UBRR0H = (unsigned char)(UBRR_VALUE >> 8);
    UBRR0L = (unsigned char)UBRR_VALUE;
    
    // 2. Enable Transmitter (TX) only. We only send data to PC, not receive.
    UCSR0B = (1 << TXEN0); 
    
    // 3. Set Frame format: 8 data bits, 1 stop bit (8N1)
    // UCSZ01 and UCSZ00 set to 1 means 8-bit character size.
    UCSR0C = (1 << UCSZ01) | (1 << UCSZ00);
}

void setup_adc() {
    // 1. Select Reference Voltage and Channel
    // REFS0 = 1: Use AVcc (5V) as reference voltage.
    // MUX3..0 = 0000 (Default): Select Analog Channel 0 (A0).
    ADMUX = (1 << REFS0); 
    
    // 2. Enable ADC and set Prescaler to 128 (16MHz / 128 = 125kHz)
    // ADC clock must be between 50kHz and 200kHz for maximum 10-bit resolution.
    ADCSRA = (1 << ADEN) | (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);
}

uint16_t read_adc_raw() {
    // 1. Start Conversion by setting the ADSC (ADC Start Conversion) bit to 1.
    ADCSRA |= (1 << ADSC); 
    
    // 2. Wait for conversion to complete (Blocking / Polling)
    // The hardware clears the ADSC bit back to 0 when the conversion finishes.
    while (ADCSRA & (1 << ADSC)); 
    
    // 3. Return the 10-bit result directly from the ADC Data Register.
    return ADC; 
}

void uart_tx(char data) {
    // Wait for empty transmit buffer by checking the UDRE0 flag.
    while (!(UCSR0A & (1 << UDRE0)));
    // Put data into the buffer, which triggers the transmission.
    UDR0 = data;
}

// Custom function to convert an integer to a string and send it via UART
// Used to replace the heavy Serial.print() library function.
void uart_print_num(uint16_t n) {
    char buf[10];
    int i = 0;
    
    if (n == 0) { uart_tx('0'); return; }
    
    // Extract digits in reverse order
    while (n > 0) {
        buf[i++] = (n % 10) + '0';
        n /= 10;
    }
    
    // Send digits in correct order
    for (; i > 0; i--) uart_tx(buf[i-1]);
}

// --- Standard Arduino Entry Points ---

void setup() {
    // Bare-metal hardware initialization (No Arduino Core Libraries)
    setup_uart();
    setup_adc();
}

void loop() {
    // 1. Read the raw analog value (0 ~ 1023) from the Potentiometer
    uint16_t val = read_adc_raw();
    
    // 2. Transmit data using the Protocol: "S:<value>\n"
    uart_tx('S');
    uart_tx(':');
    uart_print_num(val);
    uart_tx('\n'); // Newline character serves as the delimiter for the PC parser
    
    // 3. Prevent serial flooding and give the PC time to process
    _delay_ms(10); 
}
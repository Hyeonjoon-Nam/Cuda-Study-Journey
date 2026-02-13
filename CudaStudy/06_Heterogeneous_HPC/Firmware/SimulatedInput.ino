int simulatedValue = 0;
int increment = 10;

void setup() {
  Serial.begin(9600);
}

void loop() {
  simulatedValue = simulatedValue + increment;

  if (simulatedValue >= 1023) {
    simulatedValue = 1023;
    increment = -increment;
  }
  else if (simulatedValue <= 0) {
    simulatedValue = 0;
    increment = -increment;
  }

  Serial.println(simulatedValue);

  delay(50);
}

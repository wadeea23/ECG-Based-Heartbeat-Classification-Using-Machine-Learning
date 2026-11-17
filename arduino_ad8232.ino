
void setup() {
  Serial.begin(115200);
  pinMode(10, INPUT);
  pinMode(11, INPUT);
}

void loop() {
  int ecg = analogRead(A0);
  int loPlus = digitalRead(10);
  int loMinus = digitalRead(11);

  Serial.print(ecg);
  Serial.print(",");
  Serial.print(loPlus);
  Serial.print(",");
  Serial.println(loMinus);

  delay(2);
}

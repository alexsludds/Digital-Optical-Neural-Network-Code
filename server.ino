int incomingByte = 0;
int camera_trigger_PIN = 1;
int blue_channel_mod_PIN = 21;
int red_channel_mod_PIN = 20;
int ledPIN = 13;
 
void setup(){
  Serial.begin(9600);
  pinMode(camera_trigger_PIN, OUTPUT);
  pinMode(ledPIN, OUTPUT);
  pinMode(blue_channel_mod_PIN,OUTPUT);
  pinMode(red_channel_mod_PIN,OUTPUT); 
}
 
void loop(){
  //Here we set the LED intensities
  //By default the analog write resolution is 8-bit, so intensity is 0-255
  int red_channel_intensity = 128;
  int blue_channel_intensity = 128;

  analogWrite(red_channel_mod_PIN,red_channel_intensity);
  analogWrite(blue_channel_mod_PIN,blue_channel_intensity);

  if (Serial.available() > 0) {
  // read the incoming byte:
  incomingByte = Serial.read();

  if(incomingByte == 48){ //ASCII printable character: 48 means number 0 as string
    //Here we just have to send a trigger signal to the camera.
    digitalWrite(camera_trigger_PIN, HIGH);
    digitalWrite(ledPIN, HIGH);
    delay(5);
    digitalWrite(camera_trigger_PIN,LOW);
    digitalWrite(ledPIN, LOW);
  }

  else if (incomingByte == 82)
  {
    //here we have received R over serial signally that we are about to receiver an integer for red channel 
    int red_int = Serial.parseInt();
    analogWrite(red_channel_mod_PIN,red_int);
  }

  else if (incomingByte == 66)
  {
    //here we receiver B over serial, signalling we are about to receiver a new intensity for the blue channel
    int blue_int = Serial.parseInt();
    analogWrite(blue_channel_mod_PIN,blue_int);
  }
  }
}

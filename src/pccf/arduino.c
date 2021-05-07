#include <DHT.h>

#define DHTPIN 2     // what pin we're connected to
#define DHTTYPE DHT22   // DHT 22  (AM2302)
DHT dht(DHTPIN, DHTTYPE); //// Initialize DHT sensor for normal 16mhz Arduino

int chk;
float hum;
float temp;

void setup()
{
  Serial.begin(9600);
  dht.begin();
}

void loop()
{
    delay(30000);
    hum = dht.readHumidity();
    temp = dht.readTemperature();
    Serial.print(temp);
    Serial.print(",  ");
    Serial.print(hum);
    Serial.print("\n");
    delay(30000);
}


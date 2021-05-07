import serial
import csv
import time

ser = serial.Serial('/dev/ttyACM0')

if __name__ == "__main__":
    print("Start reading")
    while True:
        try:
            ser_bytes = ser.readline()
            decoded_bytes = ser_bytes.decode("utf-8")
            temp, hum = decoded_bytes.split(',')
            temp = float(temp)
            hum = float(hum)
            print(f"{time.time()}, T = {temp}, Hum. = {hum}")
            with open("./temperature_data1.csv", "a") as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerow([time.time(), temp, hum])
        except KeyboardInterrupt:
            print("Keyboard Interrupt")
            break

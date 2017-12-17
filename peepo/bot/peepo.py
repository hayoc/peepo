#!/usr/bin/env python3

import ev3dev.ev3 as ev3


class Peepo:

    def __init__(self):
        self.steering = ev3.MediumMotor('outB')
        self.driving = ev3.LargeMotor('outA')
        self.touching = ev3.TouchSensor(); assert self.touching.connected
        self.infrared = ev3.InfraredSensor(); assert self.infrared.connected
        self.infrared.mode = 'IR-PROX'

    def vision(self):
        return self.infrared.value()

    def touch(self):
        return self.touching.value()

    def drive(self, speed):
        self.driving.run_forever(speed_sp=2000 * speed - 1000)

    def steer(self, degrees):
        self.steering.run_to_rel_pos(position_sp = 180 * degrees - 90, speed_sp=900, stop_action="brake")

    def stop(self):
        self.driving.stop(stop_action="brake")

    def shutdown(self):
        self.steering.stop()
        self.driving.stop()
        self.steering.run_forever(speed_sp=0)
        self.driving.run_forever(speed_sp=0)


def main():
    peepo = Peepo()


if __name__ == "__main__":
    main()

import ev3dev.ev3 as ev3

right_motor = ev3.LargeMotor('outA')
left_motor = ev3.LargeMotor('outD')

right_motor.stop()
left_motor.stop()
right_motor.run_forever(speed_sp=0)
left_motor.run_forever(speed_sp=0)

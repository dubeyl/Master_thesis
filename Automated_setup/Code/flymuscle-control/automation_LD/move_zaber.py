            
from zaber_motion import Units
from zaber_motion.ascii import Connection
#update serial port name, here "/dev/zaber_device"
with Connection.open_serial_port("/dev/zaber_device") as connection:
    connection.enable_alerts()

    device_list = connection.detect_devices()
    print("Found {} zaber devices".format(len(device_list)))
    device = device_list[0]
    axis = device.get_axis(1)

    #prepare stage for first recording
    if not axis.is_homed():
        axis.home()
    #### Move to 2mm --> to update to correct value
    axis.move_absolute(44.5, Units.LENGTH_MILLIMETRES)
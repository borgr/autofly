import time
import mpl_toolkits.basemap.pyproj as pyproj

logFile = None
vehicle = None
# cummoinication rate. 115200 for usb connection. 57600 for radio.
BAUDRATE = 115200
# note that the port might change based on on usb input u choose
IS_LINUX = True
if IS_LINUX:
    VEHICLE_PORT = r'/dev/ttyACM0'
else:
    import admin
    if not admin.isUserAdmin():
        admin.runAsAdmin()
    VEHICLE_PORT = r'com5'
DEFAULT_AIRSPEED = 1  # in meters/second


def initCopter(logFilePath, mpstate):
    """ initializing the copter. CALL THIS BEFORE FLIGHT"""
    global logFile
    logFile = logFilePath
    time.sleep(10)


def land(mpstate):
    """ switches the mode to land"""
    mpstate.send_command("mode land")


def logAction(message):
    """ logs given massage in a logFile, for later use.
    """
    outFile = open(logFile, 'a')
    outFile.write(message + "\n")
    outFile.close()
    print(message)


def arm(mpstate):
    logAction("arm() called")
#    while not vehicle.is_armable:
#        logAction('Waiting for vehicle to initialise...')
    time.sleep(15)
    logAction('Arming motors')
    mpstate.send_command("arm throttle")
    while not mpstate.armed:
        logAction('Waiting for vehicle to arm...')
        time.sleep(1)
    logAction('ARMED!')


def disarm(mpstate):
    logAction("disarm() called")
    while mpstate.armed:
        logAction('Waiting for disarming...')
        time.sleep(1)
    logAction('DISARMED')


def guided(mpstate):
    logAction('guided mode')
    mpstate.send_command("mode guided")

def takeoff(aTargetAltitude, mpstate):
    logAction('Taking off!')
    # Take off to target altitude
    mpstate.send_command("takeoff" + str(aTargetAltitude))
    # Wait until the vehicle reaches a safe height before processing the goto
    # (otherwise the command
    #  after Vehicle.commands.takeoff will execute immediately).
#    while True:
#        logAction(" Altitude: " +
#                  str(vehicle.location.global_relative_frame.alt))
#        # Trigger just below target alt.
#        if vehicle.location.global_relative_frame.alt >= aTargetAltitude*0.95:
#            logAction("Reached target altitude")
#            break
#        time.sleep(1)


def throttle(num, mpstate):
    mpstate.send_command("rc 3 " + str(num))


def yaw(num, mpstate):
    mpstate.send_command("rc 4 " + str(num))


def roll(num, mpstate):
    mpstate.send_command("rc 1 " + str(num))


def pitch(num, mpstate):
    mpstate.send_command("rc 2 " + str(num))


def fly_to(x, y, z, mpstate):
    """ flies x meters to the right, y meters forward, and z meters up """
    logAction("trying to fly to: (" +
              str(x) + "," + str(y) + "," + str(z) + ")")
    wgs84 = pyproj.Proj("+init=EPSG:4326")
    isn2004 = pyproj.Proj("+proj=lcc +lat_1=64.25 +lat_2=65.75 +lat_0=65" +
                          " +lon_0=-19 +x_0=1700000 +y_0=300000 +no_defs " +
                          "+a=6378137 +rf=298.257222101 +to_meter=1")
    current_loc = vehicle.location.global_frame
    old_x, old_y, old_z = pyproj.transform(wgs84, isn2004,
                                           current_loc.lon, current_loc.lat,
                                           current_loc.alt)
    new_x = old_x + x
    new_y = old_y + y
    new_z = old_z + z
    new_lon, new_lat, new_alt =  pyproj.transform(isn2004, wgs84, new_x, new_y, new_z)
    point = LocationGlobalRelative(new_lon, new_lat, new_alt)
    vehicle.simple_goto(point, groundspeed=DEFAULT_AIRSPEED)

def getLocation(mpstate):
    """ returns a struct containing the location (.lon, .lat, .alt) """
    return vehicle.location.global_frame

##print "Going to first point..."
##point1 = LocationGlobalRelative(-35.361354, 149.165218, 20)
##vehicle.commands.goto(point1)

### sleep so we can see the change in map
##time.sleep(10)
##arm_and_takeoff(3)
##logAction("Returning to Launch")
##print "Returning to Launch"
##vehicle.mode    = VehicleMode("RTL")
##vehicle.close()

##print "Going to second point..."
##point2 = LocationGlobalRelative(-35.363244, 149.168801, 20)
##vehicle.commands.goto(point2)

# sleep so we can see the change in map
##time.sleep(30)




#Close vehicle object before exiting script
##print "Close vehicle object"


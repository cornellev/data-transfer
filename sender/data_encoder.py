import io
from schemas import sensor_data as mavlink 
    
# Creates a MAVLink connection w/ a dummy IO stream 
mav = mavlink.MAVLink(io.BytesIO())

# Encodes data into a MAVLink message and
def encode_data (data: dict):
    """
    Encodes sensor data into a MAVLink message and returns bytes.

    Args:
        data (dict): Dictionary w/ keys as the different sensors 

    Returns:
        bytes: Encoded MAVLink packet 
    """

    gps_lat = data.get("gps_lat", 0.0)
    gps_long = data.get("gps_long", 0.0)
    left_rpm = data.get("left_rpm", 0)
    right_rpm = data.get("right_rpm", 0)
    x_accel = data.get("x_accel", 0.0)
    y_accel = data.get("y_accel", 0.0)
    z_accel = data.get("z_accel", 0.0)
    temp = data.get("temp", 0.0)

    # Pack into a SENSOR_DATA MAVLink message
    msg = mav.sensor_data_encode(
        gps_lat,
        gps_long,
        left_rpm,
        right_rpm,
        x_accel,
        y_accel,
        z_accel,
        temp
    )

    # Pack into bytes
    encoded_bytes = msg.pack(mav)

    return encoded_bytes


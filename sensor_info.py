sensor_definitions = {
    "Sensor_1": "Measures exhaled breath temperature.",
    "Sensor_2": "Detects oxygen levels in breath.",
    "Sensor_3": "Measures carbon dioxide concentration.",
    "Sensor_4": "Detects nitrogen compounds in breath.",
    "Sensor_5": "Monitors volatile organic compounds (VOCs).",
    "Sensor_6": "Detects methane and other hydrocarbon gases.",
    "Sensor_7": "Measures exhaled breath humidity.",
    "Sensor_8": "Detects sulfur compounds indicating lung function."
}

def interpret_sensor_data(sensor_data):
    interpretations = {
        sensor: f"{desc} Reading: {value}"
        for sensor, value in sensor_data.items()
        for key, desc in sensor_definitions.items() if sensor == key
    }
    return interpretations

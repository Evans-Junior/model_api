sensor_definitions = {
    "sensor_1": "Measures exhaled breath temperature.",
    "sensor_2": "Detects oxygen levels in breath.",
    "sensor_3": "Measures carbon dioxide concentration.",
    "sensor_4": "Detects nitrogen compounds in breath.",
    "sensor_5": "Monitors volatile organic compounds (VOCs).",
    "sensor_6": "Detects methane and other hydrocarbon gases.",
    "sensor_7": "Measures exhaled breath humidity.",
    "sensor_8": "Detects sulfur compounds indicating lung function."
}

def interpret_sensor_data(sensor_data):
    interpretations = {
        sensor: f"{desc} Reading: {value}"
        for sensor, value in sensor_data.items()
        for key, desc in sensor_definitions.items() if sensor == key
    }
    return interpretations

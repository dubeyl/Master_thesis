#include "utils_serial.h"

using namespace std;
using namespace boost::asio;
namespace motion = zaber::motion;


void send_command(serial_port& port, const std::string& command){
    write(port, buffer(command + "\n"));
}

bool get_is_kinematic(serial_port& port){
    unsigned char latest_byte;
    read(port, buffer(&latest_byte, 1));
    if (latest_byte == 0x01){
        return true;
    }else{
        return false;
    }
}

std::string read_response(serial_port& port) {
    boost::asio::streambuf buf;
    read_until(port, buf, "\n");
    istream is(&buf);
    string response;
    getline(is, response);
    return response;
}

void waitForArduinoReady(io_service &io, serial_port &port, std::string message) {
    char c;
    std::string response;
    while (true) {
        read(port, buffer(&c, 1));
        if (c == '\n') {
            if (response.find(message) != std::string::npos) {
                std::cout << response << std::endl;
                break;
            }
            response.clear();
        } else {
            response += c;
        }
    }
}

void sendBoolean(serial_port &serialPort, bool value) {
    char boolChar = value ? '1' : '0';
    write(serialPort, buffer(&boolChar, 1));
}

void setupArduino(io_service &io, serial_port &port, ArduinoParameters arduino_parameters) {\

    //Wait until the arduino is ready
    waitForArduinoReady(io, port, "Arduino Ready");

    // Sending start command with FPS and duration
    std::string start_command = "setup";
    send_command(port, start_command);

    waitForArduinoReady(io, port, "Waiting");
    send_command(port, std::to_string(arduino_parameters.fps) + "*");
    
    waitForArduinoReady(io, port, "Received");
    send_command(port, std::to_string(arduino_parameters.record_time) + "*");

    waitForArduinoReady(io, port, "Received");
    sendBoolean(port, arduino_parameters.alternating);

    waitForArduinoReady(io, port, "Muscle");
    sendBoolean(port, arduino_parameters.optogenetics);

    waitForArduinoReady(io, port, "Opto");
    if (arduino_parameters.optogenetics) {
        send_command(port, std::to_string(arduino_parameters.off1) + "*");
        waitForArduinoReady(io, port, "OFF1");
        send_command(port, std::to_string(arduino_parameters.on1) + "*");
        waitForArduinoReady(io, port, "ON1");
        send_command(port, std::to_string(arduino_parameters.off2) + "*");
        waitForArduinoReady(io, port, "OFF2");
    }
    cout<<"Arduino connected and ready"<<endl;
}

void stopArduino(serial_port &port) {
    // Send stop command to Arduino
    string stop_command = "stop";
    send_command(port, stop_command);
    string response;
    response = read_response(port);
    cout << "Arduino response to stop: " << response << endl;
}

void setupZaber(motion::ascii::Connection& connection, std::unique_ptr<motion::ascii::Axis>& main_axis){
    // Translation stage setup
    connection.enableAlerts();
    
    vector<motion::ascii::Device> device_list = connection.detectDevices();
    cout << "Found " << device_list.size() << " motion devices" << endl;
    motion::ascii::Device device = device_list[0];
    try {
        main_axis = make_unique<motion::ascii::Axis>(device.getAxis(1));
    } catch (const std::exception& e) {
        cerr << "Failed to initialize axis: " << e.what() << endl;
        return;
    }

    cout << "Homing translation stage" << endl;
    try {
        main_axis->home(true);
    } catch (const std::exception& e) {
        cerr << "Failed to home axis: " << e.what() << endl;
        return;
    }
    cout<<"Zaber stage connected and ready"<<endl;
}
#ifndef UTILS_SERIAL_H
#define UTILS_SERIAL_H

#include <boost/asio.hpp>
#include <string>
#include <iostream>
#include "utils.h"

#include <zaber/motion/ascii.h>


void send_command(boost::asio::serial_port& port, const std::string& command);
bool get_is_kinematic(boost::asio::serial_port& port);
std::string read_response(boost::asio::serial_port& port);
void waitForArduinoReady(boost::asio::io_service &io, boost::asio::serial_port &port, std::string message);

void setupArduino(boost::asio::io_service &io, boost::asio::serial_port &port, ArduinoParameters arduino_parameters);
void stopArduino(boost::asio::serial_port &port);

void setupZaber(zaber::motion::ascii::Connection& connection, std::unique_ptr<zaber::motion::ascii::Axis>& main_axis);



#endif // UTILS_SERIAL_H

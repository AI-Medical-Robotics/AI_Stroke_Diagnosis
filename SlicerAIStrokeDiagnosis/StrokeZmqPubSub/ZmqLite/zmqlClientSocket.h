#pragma once

void CreateZmqServerSocket() {
    // Initialize ZMQ context
    zmq::context_t context(1);

    // Create a ZMQ server socket (ZMQ_REP: reply socket for a server)
    zmq::socket_t server_socket(context, ZMQ_REP);
    server_socket.bind("tcp://*:5555");


}
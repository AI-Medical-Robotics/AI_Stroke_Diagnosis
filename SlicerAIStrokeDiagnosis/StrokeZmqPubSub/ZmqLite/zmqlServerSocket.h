#pragma once

void CreateZmqClientSocket() {
    // Initialize ZMQ context
    zmq::context_t context(1);

    // Create a ZMQ client socket (ZMQ_REQ: request socket for a client)
    zmq::socket_t client_socket(context, ZMQ_REQ);
    client_socket.connect("tcp://localhost:5555");
}
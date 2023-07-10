#pragma once

// OpenIGTLink's igtlServerSocket.h and igtlClientSocket.h
// can be done in ZMQ using zmq.hpp via zmq::socket_t class
#include <zmq.hpp>

#include <zmqlServerSocket.h>
#include <zmqlClientSocket.h>

// ZMQPS includes
#include "zmqpsLogicExport.h"
#include "zmqpsDevice.h"
#include "zmqpsDeviceFactory.h"
#include "zmqpsObject.h"
#include "zmqpsUtilies.h"
#include "zmqpsCommand.h"

// VTK includes
#include <vtkObject.h>
#include <vtkSmartPointer.h>
#include <vtkWeakPointer.h>

// STD includes
#include <string>
#include <map>
#include <vector>
#include <set>
#include <queue>

typedef vtkSmartPointer<class vtkMultiThreader> vtkMultiThreaderPointer;
typedef std::vector< vtkSmartPointer<zmqpsDevice> > zmqpsMessageDeviceListType;
typedef std::dequeue<zmqpsCommandPointer> zmqpsCommandDequeueType;

typedef vtkSmartPointer<class zmqpsConnector> zmqpsConnectorPointer;
typedef vtkSmartPointer<class zmqpsCircularBuffer> zmqpsCircularBufferPointer;
typedef vtkSmartPointer<class zmqpsCircularSectionBuffer> zmqpsCircularSectionBufferPointer;

enum ZMQ_CONNECTION_ROLE
{
    ZMQ_CONNECTION_ROLE_NOT_DEFINED,
    ZMQ_CONNECTION_ROLE_SERVER,
    ZMQ_CONNECTION_ROLE_CLIENT,
    ZMQ_CONNECTION_ROLE_TYPE
};

/**
A ZMQ connection over one TCP/IP port
*/

class zmqpsConnector : public vtkZMQPSObject
{
public:
};

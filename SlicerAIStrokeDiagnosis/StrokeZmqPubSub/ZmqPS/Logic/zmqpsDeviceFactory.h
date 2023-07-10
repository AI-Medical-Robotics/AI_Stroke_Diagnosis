#pragma once

#include <map>
#include <string>

// VTK includes
#include <vtkObject.h>
#include <vtkSmartPointer.h>

// ZMQPS includes
#include "zmqpsLogicExport.h"
#include "zmqpsDevice.h"

typedef vtkSmartPointer<class zmqpsDeviceCreator> zmqpsDeviceCreatorPointer;
typedef vtkSmartPointer<class zmqpsDeviceFactory> zmqpsDeviceFactoryPointer;

class ZMQPS_LOGIC_EXPORT zmqpsDeviceFactory : public vtkObject
{
public:

};

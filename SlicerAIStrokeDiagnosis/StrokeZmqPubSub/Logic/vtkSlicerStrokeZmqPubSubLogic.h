/*==============================================================================

  Program: 3D Slicer

  Portions (c) Copyright Brigham and Women's Hospital (BWH) All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

==============================================================================*/

// .NAME vtkSlicerStrokeZmqPubSubLogic - slicer logic class for volumes manipulation
// .SECTION Description
// This class manages the logic associated with reading, saving,
// and changing propertied of the volumes


#ifndef __vtkSlicerStrokeZmqPubSubLogic_h
#define __vtkSlicerStrokeZmqPubSubLogic_h

// Slicer includes
#include "vtkSlicerBaseLogic.h"
#include "vtkSlicerModuleLogic.h"

// MRML includes
#include <vtkMRMLScene.h>

// ZMQ includes
// I think includes ZMQDevice class
#include "vtkMRMLZmqConnectorNode.h"

// VTK includes
#include <vtkCallbackCommand.h>
#include <vtkMultiThreader.h>

// STD includes
#include <vector>
#include <string.h>

#include <cstdlib>

#include "vtkSlicerStrokeZmqPubSubModuleLogicExport.h"


/// \ingroup Slicer_QtModules_ExtensionTemplate
class VTK_SLICER_STROKEZMQPUBSUB_MODULE_LOGIC_EXPORT vtkSlicerStrokeZmqPubSubLogic : public vtkSlicerModuleLogic
{
public:

  enum // Events
  {
    ZMQStatusUpdateEvent = 50001,
  };

  typedef struct
  {
    std::string name;
    std::string type;
    int io;
    std::string nodeID;
    // Add any additional fields specific to ZMQ comm if needed
  } ZMQMrmlNodeInfoType;

  typedef std::vector<ZMQMrmlNodeInfoType> ZMQMrmlNodeListType;

  static vtkSlicerStrokeZmqPubSubLogic *New();
  vtkTypeMacro(vtkSlicerStrokeZmqPubSubLogic, vtkSlicerModuleLogic);
  void PrintSelf(ostream& os, vtkIndent indent) override;



  void SetMRMLSceneInternal(vtkMRMLScene* newScene) override;
  /// Register MRML Node classes to Scene. Gets called automatically when the MRMLScene is attached to this logic class.
  void RegisterNodes() override;

  //-------
  // Events
  //-------

  void OnMRMLSceneEndImport() override;
  void UpdateFromMRMLScene() override;
  void OnMRMLSceneNodeAdded(vtkMRMLNode* node) override;
  void OnMRMLSceneNodeRemoved(vtkMRMLNode* node) override;
  void OnMRMLNodeModified(vtkMRMLNode* node) override {}

  //-------------------------------------------------------------
  // Connector and device management
  //-------------------------------------------------------------
  vtkMRMLZmqConnectorNode* GetConnector(std::vtkSetStringMacro& conID);

  // Call timer-driven routines for each connector
  void CallConnectorTimerHandler();

  // Device Name management
  int SetRestrictDeviceName(int f);

  // Define a smart pointer to custom ZMQ device
  // TODO(jg): this might be accessed in vtkMRMLZmqConnectorNode header?
  typedef vtkSmartPointer<ZMQDevice> ZMQDevicePointer;

  int RegisterMessageDevice(ZMQDevicePointer device);
  int UnregisterMessageDevice(ZMQDevicePointer device);

  unsigned int GetNumberOfDevices();
  ZMQDevicePointer GetDevice(unsigned int i);
  ZMQDevicePointer GetDeviceByMRMLTag(const char* mrmlTag);
  ZMQDevicePointer GetDeviceByDeviceType(const char* deviceType);

  //-------------------------------------------------------------
  // MRML Management
  //-------------------------------------------------------------

  virtual void ProcessMRMLNodesEvents(vtkObject* caller, unsigned long event, void* callData) override;

  void ProcCommand(const char* nodeName, int size, unsigned char* data);

  void GetDeviceNamesFromMrml(ZMQMrmlNodeListType& list);
  void GetDeviceNamesFromMrml(ZMQMrmlNodeListType& list, const char* mrmlTagName);

  // Transform locator model node reference role
  vtkGetStringMacro(LocatorModelReferenceRole);

protected:
  vtkSlicerStrokeZmqPubSubLogic();
  virtual ~vtkSlicerStrokeZmqPubSubLogic() override;

  static void DataCallback(vtkObject*, unsigned long, void*, void*);

  void AddMRMLConnectorNodeObserver(vtkMRMLZmqConnectorNode* connectorNode);
  void RemoveMRMLConnectorNodeObserver(vtkMRMLZmqConnectorNode* connectorNode);

  void RegisterMessageDevices(vtkMRMLZmqConnectorNode* connectorNode);

  void UpdateAll();
  void UpdateSliceDisplay();
  vtkCallbackCommand* DataCallbackCommand;

private:

  int Initialized;

  //-------------------------------------------------------------
  // Connector Management
  //-------------------------------------------------------------
  int RestrictDeviceName;

  char* LocatorModelReferenceRole;

  vtkSetStringMacro(LocatorModelReferenceRole);

  class vtkInternal;
  vtkInternal* Internal;

  vtkSlicerStrokeZmqPubSubLogic(const vtkSlicerStrokeZmqPubSubLogic&); // Not implemented
  void operator=(const vtkSlicerStrokeZmqPubSubLogic&); // Not implemented
};

#endif

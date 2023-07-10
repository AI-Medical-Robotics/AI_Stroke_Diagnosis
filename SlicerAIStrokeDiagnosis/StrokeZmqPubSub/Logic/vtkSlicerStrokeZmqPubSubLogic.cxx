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

// Slicer includes
#include "vtkSlicerVersionConfigure.h"
#include <vtkSlicerColorLogic.h>

// StrokeZmqPubSub MRML includes
#include "vtkMRMLZmqConnectorNode.h"
#include "vtkMRMLImageMetaListNode.h"
#include "vtkMRMLLabelMetaListNode.h"
#include "vtkMRMLZmqTrackingDataQueryNode.h"
#include "vtkMRMLZmqTrackingDataBundleNode.h"
#include "vtkMRMLZmqQueryNode.h"
#include "vtkMRMLZmqStatusNode.h"
#include "vtkMRMLZmqSensorNode.h"

// ZmqPS Device includes
  // inside ZmqPS folder
#include "zmqpsDevice.h"
#include "zmqpsConnector.h"
#include "zmqpsDeviceFactory.h"

// Zmq Proto includes
#include <zmqMedicalImageMessage.pb.h>
#include <zmqTransformMessage.pb.h>

// StrokeZmqPubSub Logic includes
#include "vtkSlicerStrokeZmqPubSubLogic.h"

// MRML includes
#include <vtkMRMLModelDisplayNode.h>
#include <vtkMRMLModelNode.h>
#include <vtkMRMLTransformNode.h>
#include <vtkMRMLScene.h>

// VTK includes
#include <vtkAppendPolyData.h>
#include <vtkCallbackCommand.h>
#include <vtkCylinderSource.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkPolyData.h>
#include <vtkObjectFactory.h>
#include <vtkSphereSource.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkIntArray.h>
#include <vtkNew.h>

// vtkAddon includes
#include <vtkStreamingVolumeCodecFactory.h>

// STD includes
#include <cassert>

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkSlicerStrokeZmqPubSubLogic);

class vtkSlicerStrokeZmqPubSubLogic::vtkInternal
{
public:
  vtkInternal(vtkSlicerStrokeZmqPubSubLogic* external);
  ~vtkInternal();

  vtkSlicerStrokeZmqPubSubLogic* External;

  // TODO(jg): implement zmqpsConnector 
  zmqpsMessageDeviceListType MessageDeviceList;
  // TODO(jg): implement zmqpsDeviceFactory
  zmqpsDeviceFactoryPointer DeviceFactory;

  // Update state of node locato model to reflect the ZMQVisible attribute of the nodes
  void SetLocatorVisibility(bool visible, vtkMRMLTransformNode* transform);
  // Add a node locator to the mrml scene
  vtkMRMLModelNode* AddLocatorModel(vtkMRMLScene* scene, std::string nodeName, double r, double g, double b);
};

//----------------------------------------------------------------------------
// vtkInternal methods
vtkSlicerStrokeZmqPubSubLogic::vtkInternal::vtkInternal(vtkSlicerStrokeZmqPubSubLogic* external)
  : External(external)
{
  // TODO(jg): implement zmqpsDeviceFactory
  this->DeviceFactory = zmqpsDeviceFactoryPointer::New();
}

vtkSlicerStrokeZmqPubSubLogic::~vtkInternal()
{

}

//----------------------------------------------------------------------------
// vtkSlicerStrokeZmqPubSubLogic methods

//----------------------------------------------------------------------------
vtkSlicerStrokeZmqPubSubLogic::vtkSlicerStrokeZmqPubSubLogic()
{
  this->Internal = new vtkInternal(this);

  // Timer Handling
  this->DataCallbackCommand = vtkCallbackCommand::New();
  this->DataCallbackCommand->SetClientData(reinterpret_cast<void*>(this));
  this->DataCallbackCommand->SetCallback(vtkSlicerStrokeZmqPubSubLogic::DataCallback);

  this->Initialized = 0;
  this->RestrictDeviceName = 0;

    // TODO(jg): implement zmqpsDeviceFactory
  std::vector<std::string> deviceTypes = this->Internal->DeviceFactory->GetAvailableDeviceTypes();
  for (size_t typeIndex = 0; typeIndex < deviceTypes.size(); typeIndex++)
  {
    this->Internal->MessageDeviceList.push_back(this->Internal->DeviceFactory->GetCreator(deviceTypes[typeIndex])->Create(""));
  }

  this->LocatorModelReferenceRole = nullptr;
  this->SetLocatorModelReferenceRole("LocatorModel");
}

//----------------------------------------------------------------------------
vtkSlicerStrokeZmqPubSubLogic::~vtkSlicerStrokeZmqPubSubLogic()
{
  if (this->DataCallbackCommand)
  {
    this->DataCallbackCommand->Delete();
  }

  if (this->GetMRMLScene())
  {
    vtkSmartPointer<vtkCollection> connectorNodes = vtkSmartPointer<vtkCollection>::Take(this->GetMRMLScene()->GetNodesByClass("vtkMRMLZmqConnectorNode"));
    for (int i = 0; i < connectorNodes->GetNumberOfItems(); ++i)
    {
      vtkMRMLZmqConnectorNode* connectorNode = vtkMRMLZmqConnectorNode::SafeDownCast(connectorNodes->GetItemAsObject(i));
      if (connectorNode)
      {
        connectorNode->Stop();
      }
    }
  }

  delete this->Internal;
}

//----------------------------------------------------------------------------
void vtkSlicerStrokeZmqPubSubLogic::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "vtkSlicerStrokeZmqPubSubLogic    " << this->GetClassName() << "\n";
}

//---------------------------------------------------------------------------
void vtkSlicerStrokeZmqPubSubLogic::SetMRMLSceneInternal(vtkMRMLScene * newScene)
{
  vtkNew<vtkIntArray> events;
  events->InsertNextValue(vtkMRMLScene::NodeAddedEvent);
  events->InsertNextValue(vtkMRMLScene::NodeRemovedEvent);
  events->InsertNextValue(vtkMRMLScene::EndBatchProcessEvent);
  this->SetAndObserveMRMLSceneEventsInternal(newScene, events.GetPointer());
}


//-----------------------------------------------------------------------------
void vtkSlicerStrokeZmqPubSubLogic::RegisterNodes()
{
  vtkMRMLScene* scene = this->GetMRMLScene();
  if (!scene)
  {
    return;
  }

  scene->RegisterNodeClass(vtkNew<vtkMRMLZmqConnectorNode>().GetPointer());
  scene->RegisterNodeClass(vtkNew<vtkMRMLImageMetaListNode>().GetPointer());
  scene->RegisterNodeClass(vtkNew<vtkMRMLLabelMetaListNode>().GetPointer());
  scene->RegisterNodeClass(vtkNew<vtkMRMLZmqTrackingDataQueryNode>().GetPointer());
  scene->RegisterNodeClass(vtkNew<vtkMRMLZmqTrackingDataBundleNode>().GetPointer());
  scene->RegisterNodeClass(vtkNew<vtkMRMLZmqStatusNode>().GetPointer());
  scene->RegisterNodeClass(vtkNew<vtkMRMLZmqSensorNode>().GetPointer());
  vtkStreamingVolumeCodecFactory* codecFactory = vtkStreamingVolumeCodecFactory::GetInstance();
}

//---------------------------------------------------------------------------
void vtkSlicerStrokeZmqPubSubLogic::DataCallback(vtkObject* vtkNotUsed(caller), 
  unsigned long vtkNotUsed(eid), void* clientData, void* vtkNotUsed(callData))
{
  vtkSlicerStrokeZmqPubSubLogic* self = reinterpret_cast<vtkSlicerStrokeZmqPubSubLogic*>(clientData);
  vtkDebugWithObjectMacro(self, "In vtkSlicerStrokeZmqPubSubLogic DataCallback");
  self->UpdateAll();
}

void vtkSlicerStrokeZmqPubSubLogic::UpdateAll()
{

}

void vtkSlicerStrokeZmqPubSubLogic::AddMRMLConnectorNodeObserver(vtkMRMLZmqConnectorNode* connectorNode)
{
  if (!connectorNode)
  {
    return;
  }
  // Making sure we dont add duplicate observation
  vtkUnObserveMRMLNodeMacro(connectorNode);

  // Start observing the connector node
  vtkNew<vtkIntArray> connectorNodeEvents;
  connectorNodeEvents->InsertNextValue(connectorNode->DeviceModifiedEvent);
  vtkObserveMRMLNodeEventsMacro(connectorNode, connectorNodeEvents.GetPointer());
}

void vtkSlicerStrokeZmqPubSubLogic::RemoveMRMLConnectorNodeObserver(vtkMRMLZmqConnectorNode* connectorNode)
{
  if (!connectorNode)
  {
    return;
  }

  vtkUnObserveMRMLNodeMacro(connectorNode);
}

void vtkSlicerStrokeZmqPubSubLogic::RegisterMessageDevices(vtkMRMLZmqConnectorNode* connectorNode)
{
  if (!connectorNode)
  {
    return;
  }

  for (unsigned short i = 0; i < this->GetNumberOfDevices(); i++)
  {
    connectorNode->AddDevice(this->GetDevice(i));
    connectorNode->ConnectEvents();
  }
}

void vtkSlicerStrokeZmqPubSubLogic::OnMRMLSceneEndImport()
{
  // Scene loading/import is finished, so now start the command processing thread
  // of all the active persistent connector nodes
  std::vector<vtkMRMLNode*> nodes;
  this->GetMRMLScene()->GetNodesByClass("vtkMRMLZmqConnectorNode", nodes);
  for (std::vector< vtkMRMLNode* >::iterator iter = nodes.begin(); iter != nodes.end(); ++iter)
  {
    vtkMRMLZmqConnectorNode* connector = vtkMRMLZmqConnectorNode::SafeDownCast(*iter);
    if (connector == nullptr)
    {
      continue;
    }

    if (connector->GetPersistent())
    {
      this->Modified();
      if (connector->GetState() != vtkMRMLZmqConnectorNode::StateOff)
      {
        connector->Start();
      }
    }
  }
}


void vtkSlicerStrokeZmqPubSubLogic::OnMRMLSceneNodeAdded(vtkMRMLNode* node)
{
  vtkMRMLZmqConnectorNode* connectorNode = vtkMRMLZmqConnectorNode::SafeDownCast(node);
  if (connectorNode)
  {
    // TODO (jg): Remove this line when the corresponding UI option will be added
    connectorNode->SetRestrictDeviceName(0);

    this->AddMRMLConnectorNodeObserver(connectorNode);
  }
}

void vtkSlicerStrokeZmqPubSubLogic::OnMRMLSceneNodeRemoved(vtkMRMLNode* node) 
{
  vtkMRMLZmqConnectorNode* connectorNode = vtkMRMLZmqConnectorNode::SafeDownCast(node);
  if (connectorNode)
  {
    this->RemoveMRMLConnectorNodeObserver(connectorNode);
  }
}

vtkMRMLZmqConnectorNode* vtkSlicerStrokeZmqPubSubLogic::GetConnector(std::vtkSetStringMacro& conID)
{
  vtkMRMLNode* node = this->GetMRMLScene()->GetNodeByID(conID);
  if (node)
  {
    vtkMRMLZmqConnectorNode* conNode = vtkMRMLZmqConnectorNode::SafeDownCast(node);
    return conNode;
  }
  else
  {
    return nullptr;
  }
}

void vtkSlicerStrokeZmqPubSubLogic::CallConnectorTimerHandler()
{
  std::vector<vtkMRMLNode*> nodes;
  this->GetMRMLScene()->GetNodesByClass("vtkMRMLZmqConnectorNode", nodes);

  std::vector<vtkMRMLNode*>::iterator iter;

  for (iter = nodes.begin(); iter != nodes.end(); iter++)
  {
    vtkMRMLZmqConnectorNode* connector = vtkMRMLZmqConnectorNode::SafeDownCast(*iter);
    if (connector == nullptr)
    {
      continue;
    }
    connector->PeriodicProcess();
  }
}

int vtkSlicerStrokeZmqPubSubLogic::SetRestrictDeviceName(int f)
{
  // making sure that f is 0 or 1
  if (f != 0)
  {
    f = 1;
  }

  this->RestrictDeviceName = f;

  std::vector<vtkMRMLNode*> nodes;
  this->GetMRMLScene()->GetNodesByClass("vtkMRMLZmqConnectorNode", nodes);

  std::vector<vtkMRMLNode*>::iterator iter;

  for (iter = nodes.begin(); iter != nodes.end(); iter++)
  {
    vtkMRMLZmqConnectorNode* connector = vtkMRMLZmqConnectorNode::SafeDownCast(*iter);
    if (connector)
    {
      connector->SetRestrictDeviceName(f);
    }
  }

  return this->RestrictDeviceName;
}

int vtkSlicerStrokeZmqPubSubLogic::RegisterMessageDevice(ZMQDevicePointer device)
{
  // TODO(jg): implement zmqpsDevice
  zmqpsDevice* Device = static_cast<zmqpsDevice*>(devicePtr);
  if (Device == nullptr)
  {
    return 0;
  }

  // Search the list and check if the same Device has already been registered
  int found = 0;

  // TODO(jg): implement zmqpsConnector 
  zmqpsMessageDeviceListType::iterator iter;
  for (iter = this->Internal->MessageDeviceList.begin();
       iter != this->Internal->MessageDeviceList.end();
       iter++)
  {
    if (Device->GetDeviceType().c_str() && (strcmp(Device->GetDeviceType().c_str(), (*iter)->GetDeviceType().c_str()) == 0))
    {
      found = 1;
    }
  }

  if (found)
  {
    return 0;
  }

  if (Device->GetDeviceType().c_str())
  {
    this->Internal->MessageDeviceList.push_back(Device);
  }
  else
  {
    return 0;
  }

  // Add the ZMQ Device to the existing connectors
  if (this->GetMRMLScene())
  {
    std::vector<vtkMRMLNode*> nodes;
    this->GetMRMLScene()->GetNodesByClass("vtkMRMLZmqConnectorNode", nodes);

    std::vector<vtkMRMLNode*>::iterator iter;
    for (iter = nodes.begin(); iter != nodes.end(); iter++)
    {
      vtkMRMLZmqConnectorNode* connector = vtkMRMLZmqConnectorNode::SafeDownCast(*iter);
      if (connector)
      {
        connector->AddDevice(Device);
      }
    }
  }

  return 1;
}

int vtkSlicerStrokeZmqPubSubLogic::UnregisterMessageDevice(ZMQDevicePointer device)
{
  zmqpsDevice* Device = static_cast<zmqpsDevice*>(devicePtr);
  if (Device == nullptr)
  {
    return 0;
  }

  // Look up the message Device list
  zmqpsMessageDeviceListType::iterator iter;
  iter = this->Internal->MessageDeviceList.begin();
  while (iter) != Device) { iter++; }

  // if the Device is on the list
  if (iter != this->Internal->MessageDeviceList.end())
  {
    this->Internal->MessageDeviceList.erase(iter);

    // Remove the Device from the existing connectors
    std::vector<vtkMRMLNode*> nodes;
    if (this->GetMRMLScene())
    {
      this->GetMRMLScene()->GetNodesByClass("vtkMRMLZmqConnectorNode", nodes);

      std::vector<vtkMRMLNode*>::iterator iter;
      for (iter = nodes.begin(); iter != nodes.end(); iter++)
      {
        vtkMRMLZmqConnectorNode* connector = vtkMRMLZmqConnectorNode::SafeDownCast(*iter);
        if (connector)
        {
          connector->RemoveDevice(Device);
        }
      }
    }

    return 1;
  }
  else
  {
    return 0;
  }
}

unsigned int vtkSlicerStrokeZmqPubSubLogic::GetNumberOfDevices()
{
  return this->Internal->MessageDeviceList.size();
}

ZMQDevicePointer vtkSlicerStrokeZmqPubSubLogic::GetDevice(unsigned int i)
{
  if (i < this->Internal->MessageDeviceList.size())
  {
    return this->Internal->MessageDeviceList[i];
  }
  else
  {
    return nullptr;
  }
}

ZMQDevicePointer vtkSlicerStrokeZmqPubSubLogic::GetDeviceByMRMLTag(const char* mrmlTag)
{
  zmqpsDevice* Device = nullptr;

  zmqpsMessageDeviceListType::iterator iter;
  for (iter = this->Internal->MessageDeviceList.begin();
       iter != this->Internal->MessageDeviceList.end();
       iter++)
  {
    if (strcmp((*iter)->GetDeviceName().c_str(), mrmlTag) == 0)
    {
      Device = *iter;
      break;
    }
  }

  return Device;
}

ZMQDevicePointer vtkSlicerStrokeZmqPubSubLogic::GetDeviceByDeviceType(const char* deviceType)
{
  zmqpsDevice* Device = nullptr;

  zmqpsMessageDeviceListType::iterator iter;
  for (iter = this->Internal->MessageDeviceList.begin();
       iter != this->Internal->MessageDeviceList.end();
       iter++)
  {
    if (strcmp((*iter)->GetDeviceName().c_str(), deviceType) == 0)
    {
      Device = *iter;
      break;
    }
  }

  return Device;
}

void vtkSlicerStrokeZmqPubSubLogic::ProcessMRMLNodesEvents(vtkObject* caller, unsigned long event, void* callData)
{
  if (caller != nullptr)
  {
    vtkSlicerModuleLogic::ProcessMRMLNodesEvents(caller, event, callData);

    vtkMRMLZmqConnectorNode* cnode = vtkMRMLZmqConnectorNode::SafeDownCast(caller);
    if (cnode && event == cnode->DeviceModifiedEvent)
    {
      // Check visibility
      int nnodes;

      // Incoming nodes
      nnodes = cnode->GetNumberOfIncomingMRMLNodes();
      for (int i = 0; i < nnodes; i++)
      {
        vtkMRMLNode* inode = cnode->GetIncomingMRMLNode(i);
        if (inode)
        {
          const char* attr = inode->GetAttribute("ZMQVisible");
          bool visible = (attr && strcmp(attr, "true") == 0);
          zmqpsDevice* device = static_cast<zmqpsDevice*>(this->GetDeviceByMRMLTag(inode->GetNodeTagName()));
          if (device)
          {
            device->SetVisibility(visible);
          }

          vtkMRMLTransformNode* transformNode = vtkMRMLTransformNode::SafeDownCast(inode);
          if (transformNode)
          {
            this->Internal->SetLocatorVisibility(visible, transformNode);
          }
        }
      }

      // TODO (jg): finish the outgoing nodes
    }
  }
}

void ProcCommand(const char* nodeName, int size, unsigned char* data)
{

}

void GetDeviceNamesFromMrml(ZMQMrmlNodeListType& list)
{

}

void GetDeviceNamesFromMrml(ZMQMrmlNodeListType& list, const char* mrmlTagName)
{

}

void SetLocatorVisibility(bool visible, vtkMRMLTransformNode* transformNode)
{

}

vtkMRMLModelNode* AddLocatorModel(vtkMRMLScene* scene, std::string nodeName, double r, double g, double b)
{

}



//---------------------------------------------------------------------------
void vtkSlicerStrokeZmqPubSubLogic::UpdateFromMRMLScene()
{
  assert(this->GetMRMLScene() != 0);
}




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

// ZMQ includes
#include <zmq.hpp>

// Qt includes
#include <QTimer>

// Slicer base includes
#include "vtkSlicerVersionConfigure.h"
#include <qSlicerCoreApplication.h>
#include <qSlicerCoreIOManager.h>
#include <qSlicerNodeWriter.h>

// TODO (jg): check if there is zmq factory
#include <zmqObjectFactoryBase.h>

// TODO (jg): StrokeZmqPubSub includes
#include "qSlicerStrokeZmqPubSubModule.h"
#include "qSlicerStrokeZmqPubSubModuleWidget.h"
#include "qSlicerTextFileReader.h"

// TODO (jg): StrokeZmqPubSub Logic includes
#include <vtkSlicerStrokeZmqPubSubLogic.h>


// StrokeZmqPubSub MRML includes
#include "vtkMRMLZmqConnectorNode.h"

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_ExtensionTemplate
class qSlicerStrokeZmqPubSubModulePrivate
{
public:
  qSlicerStrokeZmqPubSubModulePrivate();

  QTimer ImportDataAndEventsTimer;
};

//-----------------------------------------------------------------------------
// qSlicerStrokeZmqPubSubModulePrivate methods

//-----------------------------------------------------------------------------
qSlicerStrokeZmqPubSubModulePrivate::qSlicerStrokeZmqPubSubModulePrivate()
{
}

//-----------------------------------------------------------------------------
// qSlicerStrokeZmqPubSubModule methods

//-----------------------------------------------------------------------------
qSlicerStrokeZmqPubSubModule::qSlicerStrokeZmqPubSubModule(QObject* _parent)
  : Superclass(_parent)
  , d_ptr(new qSlicerStrokeZmqPubSubModulePrivate)
{
  Q_D(qSlicerStrokeZmqPubSubModule);

  connect(&d->ImportDataAndEventsTimer, SIGNAL(timeout()),
          this, SLOT(importDataAndEvents()));
  
  vtkMRMLScene* scene = qSlicerCoreApplication::application()->mrmlScene();
  if(scene)
  {
    // Need to listen for any new zmq connector nodes being added to start/stop timer
    this->qvtkConnect(scene, vtkMRMLScene::NodeAddedEvent,
                      this, SLOT(onNodeAddedEvent(vtkObject*, vtkObject*)));

    this->qvtkConnect(scene, vtkMRMLScene::NodeRemovedEvent,
                      this, SLOT(onNodeRemovedEvent(vtkObject*, vtkObject*)));
  }
}

//-----------------------------------------------------------------------------
qSlicerStrokeZmqPubSubModule::~qSlicerStrokeZmqPubSubModule()
{
}

//-----------------------------------------------------------------------------
QString qSlicerStrokeZmqPubSubModule::helpText() const
{
  return "This StrokeZmq PubSub module manages communications between 3D Slicer"
         " and other ZMQ-compliant software through the network";
}

//-----------------------------------------------------------------------------
QString qSlicerStrokeZmqPubSubModule::acknowledgementText() const
{
  return "This module was supported by SJSU Computer Engineering Department";
}

//-----------------------------------------------------------------------------
QStringList qSlicerStrokeZmqPubSubModule::contributors() const
{
  QStringList moduleContributors;
  moduleContributors << QString("James Guzman (SJSU MS AI, Medical Imaging)");
  return moduleContributors;
}

//-----------------------------------------------------------------------------
QIcon qSlicerStrokeZmqPubSubModule::icon() const
{
  return QIcon(":/Icons/StrokeZmqPubSub.png");
}

//-----------------------------------------------------------------------------
QStringList qSlicerStrokeZmqPubSubModule::categories() const
{
  return QStringList() << "SjsuAIStroke";
}

//-----------------------------------------------------------------------------
QStringList qSlicerStrokeZmqPubSubModule::dependencies() const
{
  return QStringList();
}

//-----------------------------------------------------------------------------
void qSlicerStrokeZmqPubSubModule::setup()
{
  this->Superclass::setup();

  qSlicerCoreApplication* app = qSlicerCoreApplication::application();

  // This call ensures that the initialization is called on a single thread
    // If the object factory is initialized simultaneously on multiple threads,
    // there can be a conflict between factory initializations
  // Initialize ZeroMQ context and socket
  try
  {
    zmq::context_t context(1); // Create a ZeroMQ context with 1 I/O thread
    zmq::socket_t socket(context, ZMQ_PUB); // Create a ZeroMQ publisher socket
  }
  catch(const zmq::error_r& e)
  {
    std::cerr << "Error initializing ZeroMQ: " << e.what() << std::endl;
  }
}

void qSlicerStrokeZmqPubSubModule::setMRMLScene(vtkMRMLScene* scene)
{
  vtkMRMLScene* oldScene = this->mrmlScene();
  this->Superclass::setMRMLScene(scene);

  if(scene == nullptr)
  {
    return;
  }

  // Need to listen for any new zmq connector nodes being added to start/stop timer
  this->qvtkReconnect(oldScene, scene, vtkMRMLScene::NodeAddedEvent,
                    this, SLOT(onNodeAddedEvent(vtkObject*, vtkObject*)));

  this->qvtkReconnect(oldScene, scene, vtkMRMLScene::NodeRemovedEvent,
                    this, SLOT(onNodeRemovedEvent(vtkObject*, vtkObject*)));
}

//-----------------------------------------------------------------------------
qSlicerAbstractModuleRepresentation* qSlicerStrokeZmqPubSubModule::createWidgetRepresentation()
{
  return new qSlicerStrokeZmqPubSubModuleWidget;
}

//-----------------------------------------------------------------------------
vtkMRMLAbstractLogic* qSlicerStrokeZmqPubSubModule::createLogic()
{
  return vtkSlicerStrokeZmqPubSubLogic::New();
}

void qSlicerStrokeZmqPubSubModule::onNodeAddedEvent(vtkObject*, vtkObject* node)
{
  Q_D(qSlicerStrokeZmqPubSubModule);

  vtkMRMLZmqConnectorNode* connectorNode = vtkMRMLZmqConnectorNode::SafeDownCast(node);
  if(connectorNode)
  {
    // If the timer is not active
    if(!d->ImportDataAndEventsTimer.isActive())
    {
      d->ImportDataAndEventsTimer.start(5);
    }
  }
}

void qSlicerStrokeZmqPubSubModule::onNodeRemovedEvent(vtkObject*, vtkObject* node)
{
  Q_D(qSlicerStrokeZmqPubSubModule);

  vtkMRMLZmqConnectorNode* connectorNode = vtkMRMLZmqConnectorNode::SafeDownCast(node);
  if(connectorNode)
  {
    // If the timer is active
    if(d->ImportDataAndEventsTimer.isActive())
    {
      // Check if there is any other connector node left in the Scene
      vtkMRMLScene* scene = qSlicerCoreApplication::application()->mrmlScene();
      if(scene)
      {
        std::vector<vtkMRMLNode*> nodes;
        this->mrmlScene()->GetNodesByClass("vtkMRMLZmqConnectorNode", nodes);
        if(nodes.size() == 0)
        {
          // The last connector was removed
          d->ImportDataAndEventsTimer.stop();
        }
      }
    }
  }
}

void qSlicerStrokeZmqPubSubModule::importDataAndEvents()
{
  vtkMRMLAbstractLogic* l = this->logic();
  vtkSlicerStrokeZmqPubSubLogic* zmqLogic = vtkSlicerStrokeZmqPubSubLogic::SafeDownCast(l);
  if(zmqLogic)
  {
    zmqLogic->CallConnectorTimeHandler();
  }
}

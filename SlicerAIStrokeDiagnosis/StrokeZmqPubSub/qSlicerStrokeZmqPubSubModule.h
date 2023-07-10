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

#ifndef __qSlicerStrokeZmqPubSubModule_h
#define __qSlicerStrokeZmqPubSubModule_h

// Slicer includes
#include "qSlicerLoadableModule.h"

#include "qSlicerStrokeZmqPubSubModuleExport.h"

#include "qSlicerApplication.h"

class qSlicerStrokeZmqPubSubModulePrivate;
class vtkObject;

/// \ingroup Slicer_QtModules_ExtensionTemplate
class Q_SLICER_QTMODULES_STROKEZMQPUBSUB_EXPORT qSlicerStrokeZmqPubSubModule
  : public qSlicerLoadableModule
{
  Q_OBJECT;
  QVTK_OBJECT;
  Q_PLUGIN_METADATA(IID "org.slicer.modules.loadable.qSlicerLoadableModule/1.0");
  Q_INTERFACES(qSlicerLoadableModule);

public:

  typedef qSlicerLoadableModule Superclass;
  explicit qSlicerStrokeZmqPubSubModule(QObject *parent=nullptr);
  ~qSlicerStrokeZmqPubSubModule() override;

  qSlicerGetTitleMacro(QTMODULE_TITLE);

  QString helpText()const override;
  QString acknowledgementText()const override;
  QStringList contributors()const override;

  QIcon icon()const override;

  QStringList categories()const override;
  QStringList dependencies() const override;

protected:

  /// Initialize the module. Register the volumes reader/writer
  void setup() override;

  void setMRMLScene(vtkMRMLScene* scene);

  /// Create and return the widget representation associated to this module
  qSlicerAbstractModuleRepresentation * createWidgetRepresentation() override;

  /// Create and return the logic associated to this module
  vtkMRMLAbstractLogic* createLogic() override;

public slots:
  void setMRMLScene(vtkMRMLScene* scene);
  void onNodeAddedEvent(vtkObject*, vtkObject* node);
  void onNodeRemovedEvent(vtkObject*, vtkObject* node);
  void importDataAndEvents();

protected:
  QScopedPointer<qSlicerStrokeZmqPubSubModulePrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(qSlicerStrokeZmqPubSubModule);
  Q_DISABLE_COPY(qSlicerStrokeZmqPubSubModule);

};

#endif

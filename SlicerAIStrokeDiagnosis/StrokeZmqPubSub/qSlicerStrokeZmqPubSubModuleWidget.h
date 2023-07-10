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

#ifndef __qSlicerStrokeZmqPubSubModuleWidget_h
#define __qSlicerStrokeZmqPubSubModuleWidget_h

// Slicer includes
#include "qSlicerAbstractModuleWidget.h"

#include "qSlicerStrokeZmqPubSubModuleExport.h"

class qSlicerStrokeZmqPubSubModuleWidgetPrivate;
class vtkMRMLNode;

/// \ingroup Slicer_QtModules_ExtensionTemplate
class Q_SLICER_QTMODULES_STROKEZMQPUBSUB_EXPORT qSlicerStrokeZmqPubSubModuleWidget :
  public qSlicerAbstractModuleWidget
{
  Q_OBJECT

public:

  typedef qSlicerAbstractModuleWidget Superclass;
  qSlicerStrokeZmqPubSubModuleWidget(QWidget *parent=0);
  virtual ~qSlicerStrokeZmqPubSubModuleWidget();

public slots:


protected:
  QScopedPointer<qSlicerStrokeZmqPubSubModuleWidgetPrivate> d_ptr;

  void setup() override;

private:
  Q_DECLARE_PRIVATE(qSlicerStrokeZmqPubSubModuleWidget);
  Q_DISABLE_COPY(qSlicerStrokeZmqPubSubModuleWidget);
};

#endif

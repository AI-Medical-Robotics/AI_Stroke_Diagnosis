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

// Qt includes
#include <QDebug>

// Slicer includes
#include "qSlicerStrokeZmqPubSubModuleWidget.h"
#include "ui_qSlicerStrokeZmqPubSubModuleWidget.h"

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_ExtensionTemplate
class qSlicerStrokeZmqPubSubModuleWidgetPrivate: public Ui_qSlicerStrokeZmqPubSubModuleWidget
{
public:
  qSlicerStrokeZmqPubSubModuleWidgetPrivate();
};

//-----------------------------------------------------------------------------
// qSlicerStrokeZmqPubSubModuleWidgetPrivate methods

//-----------------------------------------------------------------------------
qSlicerStrokeZmqPubSubModuleWidgetPrivate::qSlicerStrokeZmqPubSubModuleWidgetPrivate()
{
}

//-----------------------------------------------------------------------------
// qSlicerStrokeZmqPubSubModuleWidget methods

//-----------------------------------------------------------------------------
qSlicerStrokeZmqPubSubModuleWidget::qSlicerStrokeZmqPubSubModuleWidget(QWidget* _parent)
  : Superclass( _parent )
  , d_ptr( new qSlicerStrokeZmqPubSubModuleWidgetPrivate )
{
}

//-----------------------------------------------------------------------------
qSlicerStrokeZmqPubSubModuleWidget::~qSlicerStrokeZmqPubSubModuleWidget()
{
}

//-----------------------------------------------------------------------------
void qSlicerStrokeZmqPubSubModuleWidget::setup()
{
  Q_D(qSlicerStrokeZmqPubSubModuleWidget);
  d->setupUi(this);
  this->Superclass::setup();
}

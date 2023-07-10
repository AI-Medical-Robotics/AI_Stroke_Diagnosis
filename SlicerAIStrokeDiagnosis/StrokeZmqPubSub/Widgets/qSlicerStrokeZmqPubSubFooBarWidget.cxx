/*==============================================================================

  Program: 3D Slicer

  Copyright (c) Kitware Inc.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

// FooBar Widgets includes
#include "qSlicerStrokeZmqPubSubFooBarWidget.h"
#include "ui_qSlicerStrokeZmqPubSubFooBarWidget.h"

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_StrokeZmqPubSub
class qSlicerStrokeZmqPubSubFooBarWidgetPrivate
  : public Ui_qSlicerStrokeZmqPubSubFooBarWidget
{
  Q_DECLARE_PUBLIC(qSlicerStrokeZmqPubSubFooBarWidget);
protected:
  qSlicerStrokeZmqPubSubFooBarWidget* const q_ptr;

public:
  qSlicerStrokeZmqPubSubFooBarWidgetPrivate(
    qSlicerStrokeZmqPubSubFooBarWidget& object);
  virtual void setupUi(qSlicerStrokeZmqPubSubFooBarWidget*);
};

// --------------------------------------------------------------------------
qSlicerStrokeZmqPubSubFooBarWidgetPrivate
::qSlicerStrokeZmqPubSubFooBarWidgetPrivate(
  qSlicerStrokeZmqPubSubFooBarWidget& object)
  : q_ptr(&object)
{
}

// --------------------------------------------------------------------------
void qSlicerStrokeZmqPubSubFooBarWidgetPrivate
::setupUi(qSlicerStrokeZmqPubSubFooBarWidget* widget)
{
  this->Ui_qSlicerStrokeZmqPubSubFooBarWidget::setupUi(widget);
}

//-----------------------------------------------------------------------------
// qSlicerStrokeZmqPubSubFooBarWidget methods

//-----------------------------------------------------------------------------
qSlicerStrokeZmqPubSubFooBarWidget
::qSlicerStrokeZmqPubSubFooBarWidget(QWidget* parentWidget)
  : Superclass( parentWidget )
  , d_ptr( new qSlicerStrokeZmqPubSubFooBarWidgetPrivate(*this) )
{
  Q_D(qSlicerStrokeZmqPubSubFooBarWidget);
  d->setupUi(this);
}

//-----------------------------------------------------------------------------
qSlicerStrokeZmqPubSubFooBarWidget
::~qSlicerStrokeZmqPubSubFooBarWidget()
{
}

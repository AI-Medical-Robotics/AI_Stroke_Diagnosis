/*==============================================================================

  Copyright (c) The Intervention Centre
  Oslo University Hospital, Oslo, Norway. All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Rafael Palomar (The Intervention Centre,
  Oslo University Hospital) and was supported by The Research Council of Norway
  through the ALive project (grant nr. 311393).

==============================================================================*/

#ifndef __vtkSlicerLoadDataZmqMarkupslogic_h_
#define __vtkSlicerLoadDataZmqMarkupslogic_h_

#include <vtkSlicerMarkupsLogic.h>

#include "vtkSlicerLoadDataZmqModuleLogicExport.h"

class VTK_SLICER_LOADDATAZMQ_MODULE_LOGIC_EXPORT vtkSlicerLoadDataZmqLogic:
  public vtkSlicerMarkupsLogic
{
public:
  static vtkSlicerLoadDataZmqLogic* New();
  vtkTypeMacro(vtkSlicerLoadDataZmqLogic, vtkSlicerMarkupsLogic);
  void PrintSelf(ostream& os, vtkIndent indent) override;

protected:
  vtkSlicerLoadDataZmqLogic();
  ~vtkSlicerLoadDataZmqLogic() override;

  void RegisterNodes() override;

private:
  vtkSlicerLoadDataZmqLogic(const vtkSlicerLoadDataZmqLogic&) = delete;
  void operator=(const vtkSlicerLoadDataZmqLogic&) = delete;
};

#endif // __vtkSlicerLoadDataZmqMarkupslogic_h_

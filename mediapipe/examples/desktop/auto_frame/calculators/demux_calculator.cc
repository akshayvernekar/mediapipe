// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"

namespace mediapipe {

namespace {
    // inputs
    constexpr char kInputTag[] = "IN_IMG";
    constexpr char kSelectTag[] = "SELECT";

    //outputs
    constexpr char kOutput0Tag[] = "OUT_IMG0";
    constexpr char kOutput1Tag[] = "OUT_IMG1";
} 

class DemuxCalculator : public CalculatorBase {
 public:

    static absl::Status GetContract(CalculatorContract* cc)
    {
        if (cc->Inputs().HasTag(kInputTag)) {
            cc->Inputs().Tag(kInputTag).Set<ImageFrame>();
        }

        if (cc->Inputs().HasTag(kSelectTag)) {
            cc->Inputs().Tag(kSelectTag).Set<int>();
        }

        if (cc->Outputs().HasTag(kOutput0Tag)) {
            cc->Outputs().Tag(kOutput0Tag).Set<ImageFrame>();
        }

        if (cc->Outputs().HasTag(kOutput1Tag)) {
            cc->Outputs().Tag(kOutput1Tag).Set<ImageFrame>();
        }
        return absl::OkStatus();    
    }

    absl::Status Open(CalculatorContext* cc) override {
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override 
    {
        int sel = cc->Inputs().Tag(kSelectTag).Get<int>();
        if(sel == 0)
        {
            cc->Outputs().Tag(kOutput0Tag).AddPacket(cc->Inputs().Tag(kInputTag).Value());
        }
        else if (sel == 1)
        {
            cc->Outputs().Tag(kOutput1Tag).AddPacket(cc->Inputs().Tag(kInputTag).Value()); 
        }
        return absl::OkStatus();
    }

    private:
};

REGISTER_CALCULATOR(DemuxCalculator);

}  // namespace mediapipe
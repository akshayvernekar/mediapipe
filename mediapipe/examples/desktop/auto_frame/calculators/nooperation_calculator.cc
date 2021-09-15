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
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/examples/desktop/auto_frame/autoframe_messages.pb.h"

namespace mediapipe {

namespace {
    // inputs
    constexpr char kInputTag[] = "IMAGE";

    //outputs
    constexpr char kOutputTag[] = "DETECTION";
} 

class NoopCalculator : public CalculatorBase {
 public:

    static absl::Status GetContract(CalculatorContract* cc)
    {
        if (cc->Inputs().HasTag(kInputTag)) {
            cc->Inputs().Tag(kInputTag).Set<ImageFrame>();
        }

        if (cc->Outputs().HasTag(kOutputTag)) {
            cc->Outputs().Tag(kOutputTag).Set<mediapipe::CombinedDetection>();
        }

        return absl::OkStatus();    
    }

    absl::Status Open(CalculatorContext* cc) override {
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override 
    {
        std::unique_ptr<mediapipe::CombinedDetection> detection = std::make_unique<mediapipe::CombinedDetection>();
        detection->set_type(mediapipe::CombinedDetection::NONE);

        cc->Outputs()
            .Tag(kOutputTag)
            .AddPacket(Adopt(detection.release()).At(cc->InputTimestamp()));
        return absl::OkStatus();
    }

    private:
};

REGISTER_CALCULATOR(NoopCalculator);

}  // namespace mediapipe
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

#include "mediapipe/examples/desktop/auto_frame/autoframe_messages.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"

namespace mediapipe {

    namespace {
        // inputs
        constexpr char kInputTag[] = "IMAGE";
        constexpr char kPrevDetectionTag[] = "PREV_DETECTION";

        // outputs
        constexpr char kOutputTag[] = "DETECTION";
    }  // namespace

    class NoopCalculator : public CalculatorBase {
       public:
        static absl::Status GetContract(CalculatorContract* cc) {
            if (cc->Inputs().HasTag(kInputTag)) {
                cc->Inputs().Tag(kInputTag).Set<ImageFrame>();
            }

            if (cc->Inputs().HasTag(kPrevDetectionTag)) {
                cc->Inputs()
                    .Tag(kPrevDetectionTag)
                    .Set<mediapipe::CombinedDetection>();
            }

            if (cc->Outputs().HasTag(kOutputTag)) {
                cc->Outputs()
                    .Tag(kOutputTag)
                    .Set<mediapipe::CombinedDetection>();
            }

            return absl::OkStatus();
        }

        absl::Status Open(CalculatorContext* cc) override {
            return absl::OkStatus();
        }

        absl::Status Process(CalculatorContext* cc) override {
            // this calculator essentially passes through the prev detection as
            // current detection
            cc->Outputs()
                .Tag(kOutputTag)
                .AddPacket(cc->Inputs().Tag(kPrevDetectionTag).Value());
            return absl::OkStatus();
        }
    };

    REGISTER_CALCULATOR(NoopCalculator);

}  // namespace mediapipe
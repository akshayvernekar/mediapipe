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
        constexpr char kInput0Tag[] = "IN_IMG0";
        constexpr char kInput1Tag[] = "IN_IMG1";
        constexpr char kSelectTag[] = "SELECT";

        // outputs
        constexpr char kOutputTag[] = "OUT_IMG";
    }  // namespace

    class MuxCalculator : public CalculatorBase {
       public:
        static absl::Status GetContract(CalculatorContract* cc) {
            for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
                cc->Inputs().Index(i).SetSameAs(&cc->Outputs().Index(0));
            }
            if (cc->Inputs().HasTag(kInput0Tag)) {
                cc->Inputs().Tag(kInput0Tag).Set<ImageFrame>();
            }

            if (cc->Inputs().HasTag(kInput1Tag)) {
                cc->Inputs().Tag(kInput1Tag).Set<ImageFrame>();
            }

            if (cc->Inputs().HasTag(kSelectTag)) {
                cc->Inputs().Tag(kSelectTag).Set<int>();
            }

            if (cc->Outputs().HasTag(kOutputTag)) {
                cc->Outputs().Tag(kOutputTag).Set<ImageFrame>();
            }
            return absl::OkStatus();
        }

        absl::Status Open(CalculatorContext* cc) override {
            cc->SetOffset(TimestampDiff(0));
            return absl::OkStatus();
        }

        absl::Status Process(CalculatorContext* cc) override {
            LOG(ERROR) << "Process MuxCalculator";
            if (!cc->Inputs().Tag(kSelectTag).IsEmpty()) {
                int sel = cc->Inputs().Tag(kSelectTag).Get<int>();
                LOG(ERROR) << "Process MuxCalculator: sel " << sel;
                if (sel == 0) {
                    LOG(ERROR) << "Sending kInput0Tag " << sel;
                    if (!cc->Inputs().Tag(kInput0Tag).IsEmpty()) {
                        cc->Outputs()
                            .Tag(kOutputTag)
                            .AddPacket(cc->Inputs().Tag(kInput0Tag).Value());
                        return absl::OkStatus();
                    }
                } else if (sel == 1) {
                    LOG(ERROR) << "Sending kInput0Tag " << sel;
                    if (!cc->Inputs().Tag(kInput1Tag).IsEmpty()) {
                        cc->Outputs()
                            .Tag(kOutputTag)
                            .AddPacket(cc->Inputs().Tag(kInput1Tag).Value());
                        return absl::OkStatus();
                    }
                }
            }

            cc->Outputs()
                .Tag(kOutputTag)
                .AddPacket(MakePacket<ImageFrame>(ImageFormat::SRGB, 640, 480)
                               .At(cc->InputTimestamp()));
            return absl::OkStatus();
        }

       private:
    };

    REGISTER_CALCULATOR(MuxCalculator);

}  // namespace mediapipe
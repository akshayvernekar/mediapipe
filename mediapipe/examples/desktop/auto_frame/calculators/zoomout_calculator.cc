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

        const float zoom_scale = 0.9;

        struct DetectionSpec {
            absl::optional<std::pair<int, int>> image_size;
        };
    }  // namespace

    class ZoomOutCalculator : public CalculatorBase {
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
            if (cc->Inputs().HasTag(kInputTag) &&
                cc->Inputs().Tag(kInputTag).IsEmpty()) {
                return absl::OkStatus();
            }

            mediapipe::CombinedDetection prev_detection;
            if (cc->Inputs().HasTag(kPrevDetectionTag) &&
                cc->Inputs().Tag(kPrevDetectionTag).IsEmpty()) {
                return absl::OkStatus();
            } else {
                prev_detection = cc->Inputs()
                                     .Tag(kPrevDetectionTag)
                                     .Get<mediapipe::CombinedDetection>();
            }

            auto& input_frame = cc->Inputs().Tag(kInputTag).Get<ImageFrame>();
            int32 frame_width = input_frame.Width();
            int32 frame_height = input_frame.Height();

            std::unique_ptr<mediapipe::CombinedDetection> detection =
                std::make_unique<mediapipe::CombinedDetection>();
            InitCombinedDetection(detection.get(), frame_width, frame_height);

            // lets copy the mid points
            detection->set_type(mediapipe::CombinedDetection::BBOX);

            mediapipe::Rect* rect = detection->mutable_bbox();
            rect->set_x_center(prev_detection.bbox().x_center());
            rect->set_y_center(prev_detection.bbox().y_center());

            int32 final_width, final_height;
            final_width = prev_detection.bbox().width() +
                          prev_detection.bbox().width() * (1 - zoom_scale);
            final_height = prev_detection.bbox().height() +
                           prev_detection.bbox().height() * (1 - zoom_scale);
            // Max zoom will be 4x or 25% of total image size
            final_width = std::min(final_width, frame_width);
            final_height = std::min(final_height, frame_height);

            rect->set_width(final_width);
            rect->set_height(final_height);

            cc->Outputs()
                .Tag(kOutputTag)
                .AddPacket(Adopt(detection.release()).At(cc->InputTimestamp()));
            return absl::OkStatus();
        }

       private:
        void InitCombinedDetection(mediapipe::CombinedDetection* detection,
                                   int img_width, int img_height) {
            detection->set_type(mediapipe::CombinedDetection::NONE);

            auto rect = absl::make_unique<Rect>();
            rect->set_x_center(img_width / 2);
            rect->set_y_center(img_height / 2);
            rect->set_width(img_width);
            rect->set_height(img_height);

            detection->set_allocated_bbox(rect.release());
            detection->set_gesture("UNKNOWN");
        }
    };

    REGISTER_CALCULATOR(ZoomOutCalculator);

}  // namespace mediapipe
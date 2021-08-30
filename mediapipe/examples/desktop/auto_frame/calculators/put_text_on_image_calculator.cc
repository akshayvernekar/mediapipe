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

namespace 
{
    // inputs
    constexpr char kInImageTag[] = "IN_IMG";
    constexpr char kInGesturesTag[] = "GESTURE";

    //outputs
    constexpr char kOutImageTag[] = "OUT_IMG";
};


// Takes in a std::string, draws the text std::string by cv::putText(), and
// outputs an ImageFrame.
//
// Example config:
// node {
//   calculator: "PutTextOnImageCalculator"
//   input_stream: "text_to_put"
//   output_stream: "out_image_frames"
// }
// TODO: Generalize the calculator for other text use cases.
class PutTextOnImageCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Process(CalculatorContext* cc) override;
};

absl::Status PutTextOnImageCalculator::GetContract(CalculatorContract* cc) {
  if (cc->Inputs().HasTag(kInImageTag)) {
    cc->Inputs().Tag(kInImageTag).Set<ImageFrame>();
  }

  if (cc->Inputs().HasTag(kInGesturesTag)) {
    cc->Inputs().Tag(kInGesturesTag).Set<std::string>();
  }

  if (cc->Outputs().HasTag(kOutImageTag)) {
    cc->Outputs().Tag(kOutImageTag).Set<ImageFrame>();
  }

  return absl::OkStatus();
}

absl::Status PutTextOnImageCalculator::Process(CalculatorContext* cc) {

    auto& input_frame = cc->Inputs().Tag(kInImageTag).Get<ImageFrame>();
    if (!cc->Inputs().Tag(kInGesturesTag).IsEmpty())
    {
        auto& gesture_strings = cc->Inputs().Tag(kInGesturesTag).Get<std::string>();

        // Convert back to opencv for display or saving.
        cv::Mat input_frame_mat = mediapipe::formats::MatView(&input_frame);

        cv::cvtColor(input_frame_mat, input_frame_mat, cv::COLOR_RGB2BGR);

        cv::putText(input_frame_mat, gesture_strings , cv::Point(15, 70), cv::FONT_HERSHEY_PLAIN, 3,
                cv::Scalar(255, 255, 0, 255), 4);
        cv::cvtColor(input_frame_mat, input_frame_mat, cv::COLOR_BGR2RGB);

        std::unique_ptr<ImageFrame> output_frame = absl::make_unique<ImageFrame>(
            ImageFormat::SRGB, input_frame_mat.cols, input_frame_mat.rows, mediapipe::ImageFrame::kDefaultAlignmentBoundary);

        input_frame_mat.copyTo(formats::MatView(output_frame.get()));
        cc->Outputs().Tag(kOutImageTag).Add(output_frame.release(), cc->InputTimestamp());
    }
    else
    {
        cc->Outputs().Tag(kOutImageTag).AddPacket(cc->Inputs().Tag(kInImageTag).Value());
    }
    return absl::OkStatus();
}

REGISTER_CALCULATOR(PutTextOnImageCalculator);

}  // namespace mediapipe

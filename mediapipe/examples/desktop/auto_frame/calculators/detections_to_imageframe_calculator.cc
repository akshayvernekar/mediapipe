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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/examples/desktop/auto_frame/autoframe_messages.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

    namespace {

        // inputs
        constexpr char kDetectionTag[] = "DETECTION";
        constexpr char kInputImageTag[] = "IMAGE";

        // outputs
        constexpr char kOutputImageTag[] = "IMAGE";

        constexpr float kMinFloat = std::numeric_limits<float>::lowest();
        constexpr float kMaxFloat = std::numeric_limits<float>::max();

    }  // namespace

    // A calculator that converts CombinedDetection proto to Image Frame data
    //
    // CombinedDetection object could have the output of Autoframe algo or
    // Gesture recognition or Zoom
    //
    // Example config:
    // node {
    //   calculator: "DetectionsToImageframeCalculator"
    //   input_stream: "DETECTION:detection"
    //   output_stream: "IMAGE:render_image"
    // }
    class DetectionsToImageframeCalculator : public CalculatorBase {
       public:
        DetectionsToImageframeCalculator() {}
        ~DetectionsToImageframeCalculator() override {}
        DetectionsToImageframeCalculator(
            const DetectionsToImageframeCalculator&) = delete;
        DetectionsToImageframeCalculator& operator=(
            const DetectionsToImageframeCalculator&) = delete;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;

        absl::Status Process(CalculatorContext* cc) override;

       protected:
        int32 get_Euclidean_DistanceAB(int32 a_x, int32 a_y, int32 b_x,
                                       int32 b_y);
        absl::Status CropAndResizeImageCV(cv::Mat& image_frame,
                                          const mediapipe::Rect& bbox);
        absl::Status PutTextOnImageCV(cv::Mat& image_frame,
                                      const std::string gesture_strings);
    };
    REGISTER_CALCULATOR(DetectionsToImageframeCalculator);

    int32 DetectionsToImageframeCalculator::get_Euclidean_DistanceAB(
        int32 a_x, int32 a_y, int32 b_x, int32 b_y) {
        float dist = std::pow(a_x - b_x, 2) + pow(a_y - b_y, 2);
        return int32(std::sqrt(dist));
    }

    absl::Status DetectionsToImageframeCalculator::CropAndResizeImageCV(
        cv::Mat& image_frame, const mediapipe::Rect& bbox) {
        int32 image_width = image_frame.cols;
        int32 image_height = image_frame.rows;

        int32 x1 = bbox.x_center() - bbox.width() / 2;
        int32 y1 = bbox.y_center() - bbox.height() / 2;
        int32 bbox_width = bbox.width();
        int32 bbox_height = bbox.height();

        if (bbox_width == 0) bbox_width = image_width;

        if (bbox_height == 0) bbox_height = image_height;

        cv::Rect roi(x1, y1, bbox_width, bbox_height);

        cv::Mat croppedRef = image_frame(roi);

        cv::Mat croppedImage;
        croppedRef.copyTo(croppedImage);
        cv::Mat resizedImage;
        cv::resize(croppedImage, image_frame,
                   cv::Size(image_width, image_height), cv::INTER_LINEAR);

        return absl::OkStatus();
    }

    absl::Status DetectionsToImageframeCalculator::PutTextOnImageCV(
        cv::Mat& image_frame, const std::string gesture_strings) {
        cv::putText(image_frame, gesture_strings, cv::Point(15, 70),
                    cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 255, 0, 255), 4);

        return absl::OkStatus();
    }

    absl::Status DetectionsToImageframeCalculator::GetContract(
        CalculatorContract* cc) {
        RET_CHECK(cc->Inputs().HasTag(kDetectionTag))
            << "Exactly one of DETECTION or DETECTIONS input stream should be "
               "provided.";
        RET_CHECK_EQ((cc->Outputs().HasTag(kOutputImageTag) ? 1 : 0), 1)
            << "Exactly one of NORM_RECT, RECT, NORM_RECTS or RECTS output "
               "stream "
               "should be provided.";

        if (cc->Inputs().HasTag(kDetectionTag)) {
            cc->Inputs().Tag(kDetectionTag).Set<mediapipe::CombinedDetection>();
        }

        if (cc->Inputs().HasTag(kInputImageTag)) {
            cc->Inputs().Tag(kInputImageTag).Set<ImageFrame>();
        }

        if (cc->Outputs().HasTag(kOutputImageTag)) {
            cc->Outputs().Tag(kOutputImageTag).Set<mediapipe::ImageFrame>();
        }

        return absl::OkStatus();
    }

    absl::Status DetectionsToImageframeCalculator::Open(CalculatorContext* cc) {
        cc->SetOffset(TimestampDiff(0));

        return absl::OkStatus();
    }
    absl::Status DetectionsToImageframeCalculator::Process(
        CalculatorContext* cc) {
        if (cc->Inputs().HasTag(kDetectionTag) &&
            cc->Inputs().Tag(kDetectionTag).IsEmpty()) {
            cc->Outputs()
                .Tag(kOutputImageTag)
                .AddPacket(MakePacket<mediapipe::ImageFrame>().At(
                    cc->InputTimestamp()));
            return absl::OkStatus();
        }

        if (cc->Inputs().HasTag(kInputImageTag) &&
            cc->Inputs().Tag(kInputImageTag).IsEmpty()) {
            cc->Outputs()
                .Tag(kOutputImageTag)
                .AddPacket(MakePacket<mediapipe::ImageFrame>().At(
                    cc->InputTimestamp()));
            return absl::OkStatus();
        }

        auto& input_detection =
            cc->Inputs().Tag(kDetectionTag).Get<mediapipe::CombinedDetection>();
        auto& input_frame = cc->Inputs().Tag(kInputImageTag).Get<ImageFrame>();

        cv::Mat input_frame_mat = mediapipe::formats::MatView(&input_frame);
        cv::Mat input_frame_mat_cpy = input_frame_mat.clone();
        // std::string s = input_detection.DebugString();

        if (input_detection.type() == mediapipe::CombinedDetection::BBOX) {
            CropAndResizeImageCV(input_frame_mat_cpy, input_detection.bbox());
        } else if (input_detection.type() ==
                   mediapipe::CombinedDetection::GESTURE) {
            PutTextOnImageCV(input_frame_mat_cpy, input_detection.gesture());
        }

        std::unique_ptr<ImageFrame> output_frame =
            absl::make_unique<ImageFrame>(
                ImageFormat::SRGB, input_frame_mat_cpy.cols,
                input_frame_mat_cpy.rows,
                mediapipe::ImageFrame::kDefaultAlignmentBoundary);

        input_frame_mat_cpy.copyTo(formats::MatView(output_frame.get()));
        cc->Outputs()
            .Tag(kOutputImageTag)
            .Add(output_frame.release(), cc->InputTimestamp());

        // cc->Outputs().Tag(kOutputImageTag).AddPacket(cc->Inputs().Tag(kInputImageTag).Value());

        return absl::OkStatus();
    }

}  // namespace mediapipe

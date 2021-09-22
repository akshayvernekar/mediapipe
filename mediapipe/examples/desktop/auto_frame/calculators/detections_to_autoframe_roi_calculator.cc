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
#include "mediapipe/examples/desktop/auto_frame/calculators/detections_to_autoframe_roi_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

    namespace {

        // inputs
        constexpr char kDetectionsTag[] = "DETECTIONS";
        constexpr char kImageSizeTag[] = "IMAGE_SIZE";
        constexpr char kPrevBBOXTag[] = "PREV_DETECTION_BBOX";

        // outputs
        constexpr char kOutputBBOXTag[] = "DETECTION_BBOX";

        constexpr float kMinFloat = std::numeric_limits<float>::lowest();
        constexpr float kMaxFloat = std::numeric_limits<float>::max();

        template <class B, class R>
        void RectFromBox(B box, R* rect) {
            rect->set_x_center(box.xmin() + box.width() / 2);
            rect->set_y_center(box.ymin() + box.height() / 2);
            rect->set_width(box.width());
            rect->set_height(box.height());
        }

        // Dynamic options passed as calculator `input_stream` that can be used
        // for calculation of rectangle or rotation for given detection. Does
        // not include static calculator options which are available via private
        // fields.
        struct DetectionSpec {
            absl::optional<std::pair<int, int>> image_size;
        };

    }  // namespace

    class DetectionsToAutoframeROICalculator : public CalculatorBase {
       public:
        DetectionsToAutoframeROICalculator() {}
        ~DetectionsToAutoframeROICalculator() override {}
        DetectionsToAutoframeROICalculator(
            const DetectionsToAutoframeROICalculator&) = delete;
        DetectionsToAutoframeROICalculator& operator=(
            const DetectionsToAutoframeROICalculator&) = delete;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;

        absl::Status Process(CalculatorContext* cc) override;

       protected:
        int32 get_Euclidean_DistanceAB(int32 a_x, int32 a_y, int32 b_x,
                                       int32 b_y);
        virtual absl::Status DetectionToNormalizedRect(
            const std::vector<::mediapipe::Detection>& detection,
            const DetectionSpec& detection_spec,
            const ::mediapipe::Rect* prev_roi, ::mediapipe::Rect* roi);
        virtual DetectionSpec GetDetectionSpec(const CalculatorContext* cc);

        static inline float NormalizeRadians(float angle) {
            return angle -
                   2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
        }
        int start_keypoint_index_;
        int end_keypoint_index_;
        bool rotate_;
        bool output_zero_rect_for_empty_detections_;

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
    REGISTER_CALCULATOR(DetectionsToAutoframeROICalculator);

    int32 DetectionsToAutoframeROICalculator::get_Euclidean_DistanceAB(
        int32 a_x, int32 a_y, int32 b_x, int32 b_y) {
        float dist = std::pow(a_x - b_x, 2) + pow(a_y - b_y, 2);
        return int32(std::sqrt(dist));
    }

    absl::Status DetectionsToAutoframeROICalculator::DetectionToNormalizedRect(
        const std::vector<Detection>& detections,
        const DetectionSpec& detection_spec, const Rect* prev_roi, Rect* roi) {
        LocationData_RelativeBoundingBox faceUnionBBox;

        const auto& image_size = detection_spec.image_size;
        int32 frame_width = image_size->first;
        int32 frame_height = image_size->second;
        if (!detections.empty()) {
            RET_CHECK(detections[0].location_data().format() ==
                      LocationData::RELATIVE_BOUNDING_BOX)
                << "Only Detection with formats of RELATIVE_BOUNDING_BOX can "
                   "be "
                   "converted to Rect";

            // assign the first element as default element
            faceUnionBBox =
                detections[0].location_data().relative_bounding_box();

            for (int i = 1; i < detections.size(); i++) {
                LocationData_RelativeBoundingBox relative_BBox =
                    detections.at(i).location_data().relative_bounding_box();

                if (relative_BBox.xmin() < faceUnionBBox.xmin()) {
                    faceUnionBBox.set_xmin(relative_BBox.xmin());
                }
                if (relative_BBox.ymin() < faceUnionBBox.ymin()) {
                    faceUnionBBox.set_ymin(relative_BBox.ymin());
                }
                if (relative_BBox.width() > faceUnionBBox.width()) {
                    faceUnionBBox.set_width(relative_BBox.width());
                }
                if (relative_BBox.height() > faceUnionBBox.height()) {
                    faceUnionBBox.set_height(relative_BBox.height());
                }
            }

            int32 face_xmin = faceUnionBBox.xmin() * frame_width;
            int32 face_ymin = faceUnionBBox.ymin() * frame_height;
            int32 face_width = faceUnionBBox.width() * frame_width;
            int32 face_height = faceUnionBBox.height() * frame_height;

            float frame_aspect_ratio = frame_width / float(frame_height);
            // std::cout << "frame_width :" << frame_width << std::endl;
            // std::cout << "frame_height :" << frame_height << std::endl;

            // std::cout << "face_width :" << face_width << std::endl;
            // std::cout << "face_height :" << face_height << std::endl;
            // std::cout << "face_xmin :" << face_xmin << std::endl;
            // std::cout << "face_ymin :" << face_ymin << std::endl <<
            // std::endl; std::cout << "frame_aspect_ratio :" <<
            // frame_aspect_ratio << std::endl; float frame_aspect_ratio = 1;
            int32 _roi_X1 = 0;
            int32 _roi_Y1 = 0;
            int32 _roi_X2 = frame_width;
            int32 _roi_Y2 = frame_height;

            if (face_height < (frame_height / 2) &&
                face_width < (frame_width / 2)) {
                int32 final_h = std::min(
                    (int32)std::round(std::sqrt(10 * face_height * face_width /
                                                frame_aspect_ratio)),
                    frame_height);
                int32 final_w =
                    std::min((int32)std::round(frame_aspect_ratio * final_h),
                             frame_width);

                int32 x_mid = face_xmin + face_width / 2;
                int32 y_mid = face_ymin + face_height / 2;

                if (prev_roi != NULL) {
                    int32 prev_roi_xcen = prev_roi->x_center();
                    int32 prev_roi_ycen = prev_roi->y_center();

                    int32 dist = get_Euclidean_DistanceAB(
                        prev_roi_xcen, prev_roi_ycen, x_mid, y_mid);
                    if (dist < 20) {
                        *roi = *prev_roi;
                        return absl::OkStatus();
                    }
                }

                _roi_X1 = x_mid - (final_w / 2);
                if (_roi_X1 > 0) {
                    _roi_X2 = _roi_X1 + final_w;
                    if (_roi_X2 > frame_width) {
                        if ((_roi_X1 - (_roi_X2 - frame_width)) <= 0) {
                            _roi_X1 = 0;
                            _roi_X2 = frame_width;
                        } else {
                            _roi_X1 = _roi_X1 - (_roi_X2 - frame_width);
                            _roi_X2 = _roi_X1 + final_w;
                        }
                    }
                } else {
                    _roi_X1 = 0;
                    _roi_X2 = std::min(final_w, frame_width);
                }

                _roi_Y1 = y_mid - (final_h / 2);
                if (_roi_Y1 > 0) {
                    _roi_Y2 = _roi_Y1 + final_h;
                    if (_roi_Y2 > frame_height) {
                        if ((_roi_Y1 - (_roi_Y2 - frame_height)) <= 0) {
                            _roi_Y1 = 0;
                            _roi_Y2 = frame_height;
                        } else {
                            _roi_Y1 = _roi_Y1 - (_roi_Y2 - frame_height);
                            _roi_Y2 = _roi_Y1 + final_h;
                        }
                    }
                } else {
                    _roi_Y1 = 0;
                    _roi_Y2 = std::min(final_h, frame_height);
                }
            }

            int32 _roi_width = _roi_X2 - _roi_X1;
            int32 _roi_height = _roi_Y2 - _roi_Y1;
            roi->set_x_center(_roi_X1 + _roi_width / 2);
            roi->set_y_center(_roi_Y1 + _roi_height / 2);
            roi->set_width(_roi_width);
            roi->set_height(_roi_height);

        } else {
            roi->set_x_center(frame_width / 2);
            roi->set_y_center(frame_height / 2);
            roi->set_width(frame_width);
            roi->set_height(frame_height);
        }
        return absl::OkStatus();
    }

    absl::Status DetectionsToAutoframeROICalculator::GetContract(
        CalculatorContract* cc) {
        RET_CHECK(cc->Inputs().HasTag(kDetectionsTag))
            << "Exactly one of DETECTION or DETECTIONS input stream should be "
               "provided.";
        RET_CHECK_EQ((cc->Outputs().HasTag(kOutputBBOXTag) ? 1 : 0), 1)
            << "Exactly one of NORM_RECT, RECT, NORM_RECTS or RECTS output "
               "stream "
               "should be provided.";

        if (cc->Inputs().HasTag(kDetectionsTag)) {
            cc->Inputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
        }

        if (cc->Inputs().HasTag(kPrevBBOXTag)) {
            cc->Inputs().Tag(kPrevBBOXTag).Set<mediapipe::CombinedDetection>();
        }

        if (cc->Inputs().HasTag(kImageSizeTag)) {
            cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
        }

        if (cc->Outputs().HasTag(kOutputBBOXTag)) {
            cc->Outputs()
                .Tag(kOutputBBOXTag)
                .Set<mediapipe::CombinedDetection>();
        }

        return absl::OkStatus();
    }

    absl::Status DetectionsToAutoframeROICalculator::Open(
        CalculatorContext* cc) {
        cc->SetOffset(TimestampDiff(0));

        return absl::OkStatus();
    }
    absl::Status DetectionsToAutoframeROICalculator::Process(
        CalculatorContext* cc) {
        const auto& options =
            cc->Options<DetectionsToAutoframeROICalculatorOptions>();
        bool hasProduceEmptyPacket = options.produce_empty_packet();
        if (cc->Inputs().HasTag(kDetectionsTag) &&
            cc->Inputs().Tag(kDetectionsTag).IsEmpty()) {
            return absl::OkStatus();
        }

        std::vector<Detection> detections;
        if (cc->Inputs().HasTag(kDetectionsTag)) {
            detections =
                cc->Inputs().Tag(kDetectionsTag).Get<std::vector<Detection>>();

            if (detections.empty()) {
                if (hasProduceEmptyPacket) {
                    if (cc->Outputs().HasTag(kOutputBBOXTag)) {
                        cc->Outputs()
                            .Tag(kOutputBBOXTag)
                            .AddPacket(
                                MakePacket<mediapipe::CombinedDetection>().At(
                                    cc->InputTimestamp()));
                    }
                }
                return absl::OkStatus();
            }
        }

        // Get dynamic calculator options (e.g. `image_size`).
        const DetectionSpec detection_spec = GetDetectionSpec(cc);
        const auto& image_size = detection_spec.image_size;
        int32 frame_width = image_size->first;
        int32 frame_height = image_size->second;
        std::unique_ptr<mediapipe::CombinedDetection> detection =
            std::make_unique<mediapipe::CombinedDetection>();
        InitCombinedDetection(detection.get(), frame_width, frame_height);

        if (cc->Outputs().HasTag(kOutputBBOXTag)) {
            if (!cc->Inputs().Tag(kPrevBBOXTag).IsEmpty()) {
                auto prevBbox = cc->Inputs()
                                    .Tag(kPrevBBOXTag)
                                    .Get<mediapipe::CombinedDetection>();
                MP_RETURN_IF_ERROR(DetectionToNormalizedRect(
                    detections, detection_spec, &prevBbox.bbox(),
                    detection->mutable_bbox()));
            } else {
                MP_RETURN_IF_ERROR(
                    DetectionToNormalizedRect(detections, detection_spec, NULL,
                                              detection->mutable_bbox()));
            }

            detection->set_type(mediapipe::CombinedDetection::BBOX);

            cc->Outputs()
                .Tag(kOutputBBOXTag)
                .Add(detection.release(), cc->InputTimestamp());
        }

        return absl::OkStatus();
    }

    DetectionSpec DetectionsToAutoframeROICalculator::GetDetectionSpec(
        const CalculatorContext* cc) {
        absl::optional<std::pair<int, int>> image_size;
        if (HasTagValue(cc->Inputs(), kImageSizeTag)) {
            image_size =
                cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
        }

        return {image_size};
    }
}  // namespace mediapipe

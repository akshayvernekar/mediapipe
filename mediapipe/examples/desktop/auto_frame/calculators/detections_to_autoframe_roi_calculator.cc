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
constexpr char kPrevROITag[] = "PREV_ROI";

//outputs
constexpr char kNormRectTag[] = "NORM_RECT";

constexpr float kMinFloat = std::numeric_limits<float>::lowest();
constexpr float kMaxFloat = std::numeric_limits<float>::max();


template <class B, class R>
void RectFromBox(B box, R* rect) {
  rect->set_x_center(box.xmin() + box.width() / 2);
  rect->set_y_center(box.ymin() + box.height() / 2);
  rect->set_width(box.width());
  rect->set_height(box.height());
}

// Dynamic options passed as calculator `input_stream` that can be used for
// calculation of rectangle or rotation for given detection. Does not include
// static calculator options which are available via private fields.
struct DetectionSpec {
  absl::optional<std::pair<int, int>> image_size;
};


}  // namespace

// A calculator that converts Detection proto to RenderData proto for
// visualization.
//
// Detection is the format for encoding one or more detections in an image.
// The input can be std::vector<Detection> or DetectionList.
//
// Please note that only Location Data formats of BOUNDING_BOX and
// RELATIVE_BOUNDING_BOX are supported. Normalized coordinates for
// RELATIVE_BOUNDING_BOX must be between 0.0 and 1.0. Any incremental normalized
// coordinates calculation in this calculator is capped at 1.0.
//
// The text(s) for "label(_id),score" will be shown on top left
// corner of the bounding box. The text for "feature_tag" will be shown on
// bottom left corner of the bounding box.
//
// Example config:
// node {
//   calculator: "DetectionsToAutoframeROICalculator"
//   input_stream: "DETECTION:detection"
//   input_stream: "DETECTIONS:detections"
//   input_stream: "DETECTION_LIST:detection_list"
//   output_stream: "RENDER_DATA:render_data"
//   options {
//     [DetectionsToAutoframeROICalculatorOptions.ext] {
//       produce_empty_packet : false
//     }
//   }
// }
class DetectionsToAutoframeROICalculator : public CalculatorBase {
 public:
  DetectionsToAutoframeROICalculator() {}
  ~DetectionsToAutoframeROICalculator() override {}
  DetectionsToAutoframeROICalculator(const DetectionsToAutoframeROICalculator&) =
      delete;
  DetectionsToAutoframeROICalculator& operator=(
      const DetectionsToAutoframeROICalculator&) = delete;

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;

  absl::Status Process(CalculatorContext* cc) override;

 protected:
  int32 get_Euclidean_DistanceAB(int32 a_x, int32 a_y, int32 b_x, int32 b_y);
  virtual absl::Status DetectionToNormalizedRect(
      const std::vector<::mediapipe::Detection>& detection,
      const DetectionSpec& detection_spec,::mediapipe::NormalizedRect* prev_roi, ::mediapipe::NormalizedRect* roi);
  virtual DetectionSpec GetDetectionSpec(const CalculatorContext* cc);

  static inline float NormalizeRadians(float angle) {
    return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
  }
  int start_keypoint_index_;
  int end_keypoint_index_;
  bool rotate_;
  bool output_zero_rect_for_empty_detections_;
};
REGISTER_CALCULATOR(DetectionsToAutoframeROICalculator);

int32 DetectionsToAutoframeROICalculator::get_Euclidean_DistanceAB(int32 a_x, int32 a_y, int32 b_x, int32 b_y)
{
    float dist = std::pow(a_x - b_x, 2) + pow(a_y - b_y, 2);
    return int32(std::sqrt(dist));
}

absl::Status DetectionsToAutoframeROICalculator::DetectionToNormalizedRect(
    const std::vector<Detection>& detections, const DetectionSpec& detection_spec,
    NormalizedRect* prev_roi, NormalizedRect* roi) {
        LocationData_RelativeBoundingBox faceUnionBBox;
        if (!detections.empty())
        {   
            RET_CHECK(detections[0].location_data().format() == LocationData::RELATIVE_BOUNDING_BOX)
            << "Only Detection with formats of RELATIVE_BOUNDING_BOX can be "
                "converted to NormalizedRect";

            // assign the first element as default element
            faceUnionBBox = detections[0].location_data().relative_bounding_box();

            for (int i=0; i < detections.size(); i++)
            {
                LocationData_RelativeBoundingBox relative_BBox = detections.at(i).location_data().relative_bounding_box();
                
                if (relative_BBox.xmin() < faceUnionBBox.xmin())
                {
                    faceUnionBBox.set_xmin(relative_BBox.xmin()) ;
                }
                if (relative_BBox.ymin() < faceUnionBBox.ymin())
                {
                    faceUnionBBox.set_ymin(relative_BBox.ymin()) ;
                }                
                if (relative_BBox.width() > faceUnionBBox.width())
                {
                    faceUnionBBox.set_width(relative_BBox.width()) ;
                }
                if (relative_BBox.height() > faceUnionBBox.height())
                {
                    faceUnionBBox.set_height(relative_BBox.height()) ;
                }
            }
            // std::cout << "BBox : " <<faceUnionBBox.xmin() << "\t" << faceUnionBBox.ymin() << "\t" << faceUnionBBox.width() << "\t" << faceUnionBBox.height() << std::endl;
            const auto& image_size = detection_spec.image_size;
            int32 frame_width = image_size->first;
            int32 frame_height = image_size->second;

            int32 face_xmin = faceUnionBBox.xmin() * frame_width;
            int32 face_ymin = faceUnionBBox.ymin() * frame_height;
            int32 face_width = faceUnionBBox.width() * frame_width;
            int32 face_height = faceUnionBBox.height() * frame_height;

            float frame_aspect_ratio = frame_width/float(frame_height);
            // std::cout << "frame_width :" << frame_width << std::endl;
            // std::cout << "frame_height :" << frame_height << std::endl;

            // std::cout << "face_width :" << face_width << std::endl;
            // std::cout << "face_height :" << face_height << std::endl;
            // std::cout << "face_xmin :" << face_xmin << std::endl;
            // std::cout << "face_ymin :" << face_ymin << std::endl;
            // std::cout << "frame_aspect_ratio :" << frame_aspect_ratio << std::endl;
            //float frame_aspect_ratio = 1;
            int32 _roi_X1 = 0;
            int32 _roi_Y1 = 0;
            int32 _roi_width = frame_width;
            int32 _roi_height = frame_height;

            if (face_height < (frame_height/2) && face_width < (frame_width/2))
            {
                int32 final_h = (int32)std::round(std::sqrt( 10 * face_height  * face_width /  frame_aspect_ratio));
                int32 final_w = (int32)std::round(frame_aspect_ratio * final_h);
                // std::cout << "Final_h :" << final_h << "    " <<"Final_w:" << final_w << std::endl;

                float x_mid = face_xmin + face_width/2;
                float y_mid = face_ymin + face_height/2;

                _roi_X1 = x_mid - (final_w/2);
                if (_roi_X1 > 0) 
                {
                    _roi_width = final_w;
                    if (final_w + _roi_X1 > frame_width)
                    {
                        _roi_X1 = (frame_width - final_w) < 0 ?  0 : (frame_width - final_w);
                        _roi_width = std::min(final_w , frame_width);;
                    }
                }
                else
                {
                    _roi_X1 = 0;
                    _roi_width = std::min(final_w , frame_width);
                }

                _roi_Y1 = y_mid - (final_h/2);
                if (_roi_Y1 > 0) 
                {
                    _roi_height = final_h ;
                    if (_roi_height > frame_height)
                    {
                        _roi_Y1 = ((frame_height - final_h) < 0 )? 0 : (frame_height - final_h);
                        _roi_height = std::min(final_h , frame_height);
                    }
                }
                else
                {
                    _roi_height = std::min(final_h , frame_height); 
                    _roi_Y1 = 0;
                }

            }

            if(prev_roi != NULL)
            {
                int32 prev_roi_xcen = prev_roi->x_center() * frame_width;
                int32 prev_roi_ycen = prev_roi->y_center() * frame_height;

                int32 dist = get_Euclidean_DistanceAB(prev_roi_xcen, prev_roi_ycen, (_roi_X1 + _roi_width/2), (_roi_Y1 + _roi_height / 2));
                if(dist < 20)
                {
                  *roi = *prev_roi;
                  return absl::OkStatus();
                }
                // LOG(ERROR) << "Distance between prev and current ROI : "<< dist;
            }
            
            roi->set_x_center((_roi_X1 + _roi_width / 2)/float(frame_width));
            roi->set_y_center((_roi_Y1 + _roi_height / 2)/float(frame_height));
            roi->set_width(_roi_width/float(frame_width));
            roi->set_height(_roi_height/float(frame_height));

            std::cout << "RECT : " <<roi->x_center() << "\t" << roi->y_center() << "\t" << roi->width() << "\t" << roi->height() << std::endl;
        }
  return absl::OkStatus();
}


absl::Status DetectionsToAutoframeROICalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kDetectionsTag))
      << "Exactly one of DETECTION or DETECTIONS input stream should be "
         "provided.";
  RET_CHECK_EQ(( cc->Outputs().HasTag(kNormRectTag) ? 1 : 0) ,1)
      << "Exactly one of NORM_RECT, RECT, NORM_RECTS or RECTS output stream "
         "should be provided.";

  if (cc->Inputs().HasTag(kDetectionsTag)) {
    cc->Inputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
  }

  if (cc->Inputs().HasTag(kPrevROITag)) {
    cc->Inputs().Tag(kPrevROITag).Set<NormalizedRect>();
  }

  if (cc->Outputs().HasTag(kNormRectTag)) {
    cc->Outputs().Tag(kNormRectTag).Set<NormalizedRect>();
  }
  if (cc->Inputs().HasTag(kImageSizeTag)) {
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
  }

  return absl::OkStatus();
}

absl::Status DetectionsToAutoframeROICalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  return absl::OkStatus();
}
absl::Status DetectionsToAutoframeROICalculator::Process(CalculatorContext* cc) {
  const auto& options = cc->Options<DetectionsToAutoframeROICalculatorOptions>();
  bool hasProduceEmptyPacket = options.produce_empty_packet();
  if (cc->Inputs().HasTag(kDetectionsTag) &&
      cc->Inputs().Tag(kDetectionsTag).IsEmpty()) {
          cc->Outputs()
              .Tag(kNormRectTag)
              .AddPacket(MakePacket<NormalizedRect>().At(cc->InputTimestamp()));
    return absl::OkStatus();
  }

  std::vector<Detection> detections;
  if (cc->Inputs().HasTag(kDetectionsTag)) {
    detections = cc->Inputs().Tag(kDetectionsTag).Get<std::vector<Detection>>();

    if (detections.empty()) {
      if (hasProduceEmptyPacket) {
        if (cc->Outputs().HasTag(kNormRectTag)) {
          cc->Outputs()
              .Tag(kNormRectTag)
              .AddPacket(MakePacket<NormalizedRect>().At(cc->InputTimestamp()));
        }
      }
      return absl::OkStatus();
    }
  }

  // Get dynamic calculator options (e.g. `image_size`).
  const DetectionSpec detection_spec = GetDetectionSpec(cc);

  if (cc->Outputs().HasTag(kNormRectTag)) {


    auto output_rect = absl::make_unique<NormalizedRect>();

    if (!cc->Inputs().Tag(kPrevROITag).IsEmpty())
    {
      auto prevROI = cc->Inputs().Tag(kPrevROITag).Get<NormalizedRect>();
      MP_RETURN_IF_ERROR(DetectionToNormalizedRect(detections, detection_spec, &prevROI,
                                                 output_rect.get()));
      // LOG(ERROR) << "RECT : " <<prevROI.x_center() << "\t" << prevROI.y_center() << "\t" << prevROI.width() << "\t" << prevROI.height();
    }
    else
    {
      MP_RETURN_IF_ERROR(DetectionToNormalizedRect(detections, detection_spec, NULL,
                                                 output_rect.get()));

    }
    
    cc->Outputs()
        .Tag(kNormRectTag)
        .Add(output_rect.release(), cc->InputTimestamp());
  }


  return absl::OkStatus();
}

DetectionSpec DetectionsToAutoframeROICalculator::GetDetectionSpec(
    const CalculatorContext* cc) {
  absl::optional<std::pair<int, int>> image_size;
  if (HasTagValue(cc->Inputs(), kImageSizeTag)) {
    image_size = cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
  }

  return {image_size};
}
}  // namespace mediapipe

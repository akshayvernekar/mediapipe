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
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
    namespace {
        // inputs
        constexpr char kSelectTag[] = "SELECT";
        constexpr char kInputTag[] = "IN";
    }  // namespace

    // This Calculator multiplexes several input streams into a single
    // output stream, dropping input packets with timestamps older than the
    // last output packet.  In case two packets arrive with the same timestamp,
    // the packet with the lower stream index will be output and the rest will
    // be dropped.
    //
    // This Calculator optionally produces a finish inidicator as its second
    // output stream.  One indicator packet is produced for each input packet
    // received.
    //
    // This Calculator can be used with an ImmediateInputStreamHandler or with
    // the default ISH.
    //
    // This Calculator is designed to work with a Demux calculator such as
    // the RoundRobinDemuxCalculator.  Therefore, packets from different
    // input streams are normally not expected to have the same timestamp.
    //
    // NOTE: this calculator can drop packets non-deterministically, depending
    // on how fast the input streams are fed. In most cases, MuxCalculator
    // should be preferred. In particular, dropping packets can interfere with
    // rate limiting mechanisms.
    class GenericDemuxCalculator : public CalculatorBase {
       public:
        // This calculator combines any set of input streams into a single
        // output stream.  All input stream types must match the output stream
        // type.
        static absl::Status GetContract(CalculatorContract* cc);

        // Passes any input packet to the output stream immediately, unless the
        // packet timestamp is lower than a previously passed packet.
        absl::Status Process(CalculatorContext* cc) override;
        absl::Status Open(CalculatorContext* cc) override;
    };
    REGISTER_CALCULATOR(GenericDemuxCalculator);

    absl::Status GenericDemuxCalculator::GetContract(CalculatorContract* cc) {
        if (cc->Inputs().HasTag(kSelectTag)) {
            cc->Inputs()
                .Tag(kSelectTag)
                .Set<mediapipe::CombinedDetection::OP_TYPE>();
        }
        if (cc->Inputs().HasTag(kInputTag)) {
            cc->Inputs().Tag(kInputTag).SetAny();
        }
        for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
            cc->Outputs().Index(i).SetSameAs(&cc->Inputs().Tag(kInputTag));
        }
        return absl::OkStatus();
    }

    absl::Status GenericDemuxCalculator::Open(CalculatorContext* cc) {
        cc->SetOffset(TimestampDiff(0));
        return absl::OkStatus();
    }

    absl::Status GenericDemuxCalculator::Process(CalculatorContext* cc) {
        mediapipe::CombinedDetection::OP_TYPE op_sel =
            cc->Inputs()
                .Tag(kSelectTag)
                .Get<mediapipe::CombinedDetection::OP_TYPE>();

        int sel = (int)op_sel;
        cc->Outputs().Index(sel).AddPacket(cc->Inputs().Tag(kInputTag).Value());
        return absl::OkStatus();
    }

}  // namespace mediapipe

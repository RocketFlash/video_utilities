from collections import defaultdict
from dataclasses import asdict
import json


class VideoResultsAggregator:
    def aggregate(self, results):
        frame_idxs = []
        scene_ids = []
        timestamps = []
        all_outputs = defaultdict(list)

        for frame_result in results:
            frame_idxs.append(frame_result.frame_idx)
            scene_ids.append(frame_result.scene_id)
            timestamps.append(frame_result.timestamp)
            for k, v in frame_result.outputs.items():
                all_outputs[k].append(v)
        all_outputs = dict(all_outputs)

        return dict(
            frame_idxs=frame_idxs,
            scene_ids=scene_ids,
            timestamps=timestamps,
            outputs=all_outputs
        )


    def to_json(
        self,
        outputs
    ):
        outputs_processed = []
        for x in outputs:
            outputs_processed.append(asdict(x))

        output_json = json.dumps(outputs_processed, indent=4)

        return output_json


    def __call__(
        self,
        outputs
    ):
        return self.aggregate(outputs)
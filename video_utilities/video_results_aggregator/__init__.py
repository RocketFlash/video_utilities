from collections import defaultdict


class VideoResultsAggregator:
    def aggregate(self, outputs):
        return outputs


    def aggregate(self, outputs):
        results = defaultdict(list)

        for frame_outputs in outputs:
            for k, v in frame_outputs.items():
                results[k].append(v)

        return dict(results)


    def __call__(
        self,
        outputs
    ):
        return self.aggregate(outputs)
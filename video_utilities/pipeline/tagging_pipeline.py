import json
from pathlib import Path


class TaggingPipeline:
    def __init__(
        self,
        video_frame_splitter,
        video_scene_detector,
        video_captioner,
        video_results_aggregator
    ):
        self.video_frame_splitter = video_frame_splitter
        self.video_scene_detector = video_scene_detector
        self.video_captioner = video_captioner
        self.video_results_aggregator = video_results_aggregator

    def __call__(
        self,
        video_path,
        n_frames_per_scene = 1,
        expected_output_type='dict[list[str]]',
        save_dir = None
    ):
        scene_list = self.video_scene_detector(video_path)

        video_frames_data = self.video_frame_splitter(
            video_path,
            scene_list=scene_list,
            n_frames_per_scene=n_frames_per_scene
        )

        frames_results = self.video_captioner(
            video_frames_data.frames,
            expected_output_type=expected_output_type
        )

        video_results = self.video_results_aggregator.to_json(frames_results)
        if save_dir is not None:
            video_name = Path(video_path).stem
            save_path = Path(save_dir) / f"{video_name}.json"
            
            with open(save_path, "w") as outfile:
                outfile.write(video_results)

        return video_results
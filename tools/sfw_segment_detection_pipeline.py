import os
import cv2
import math
import glob
import click
import time
import timm
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
import multiprocessing as mp
from transformers import pipeline
from transformers import (
    AutoProcessor, 
    FocalNetForImageClassification
)
from tqdm.auto import tqdm
from pathlib import Path
from video_utilities import (
    VideoDownloader
)
from video_utilities import (
    VideoFrameSplitter,
    VideoFrameSplitterConfig,
)
from video_utilities.visualization import add_text_to_image
from pathlib import Path
import onnxruntime
import nudenet
from nudenet import NudeDetector


NSFW_LABELS_DETECTOR = [
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
]


class NSFWClassifier:
    VIT_BASED_CLASSIFIERS = [
        'adamcodd', 
        'falconsai', 
        'luke', 
        'perry',
        'umair'
    ]
    
    def __init__(
        self,
        model_name,
        device='cuda'
    ):
        self.device = device
        self.model_name = model_name
        model, processor = self.get_model_and_processor(
            model_name=self.model_name
        )
        self.model = model
        self.processor = processor

    
    def get_model_and_processor(
        self,
        model_name: str,
    ):
        
        if model_name in self.VIT_BASED_CLASSIFIERS:
            model_name_to_model_id = {
                'adamcodd' : "AdamCodd/vit-base-nsfw-detector",
                'falconsai' : "Falconsai/nsfw_image_detection",
                'luke' : "LukeJacob2023/nsfw-image-detector",
                'perry' : "perrytheplatypus/falconsai-finetuned-nsfw-detect",
                'umair' : "umairrkhn/fine-tuned-nsfw-classification"
            }
            model = pipeline(
                "image-classification", 
                model=model_name_to_model_id[model_name],
                device=self.device
            )
            processor = None
        elif model_name=='marqo':
            model = timm.create_model(
                "hf_hub:Marqo/nsfw-image-detection-384", 
                pretrained=True
            )
            model = model.eval()
            model = model.to(self.device)
            
            data_config = timm.data.resolve_model_data_config(model)
            processor = timm.data.create_transform(**data_config, is_training=False)
        else:
            model_id = 'lovetillion/nsfw-image-detection-large'
            processor = AutoProcessor.from_pretrained(
                model_id
            )
            model = FocalNetForImageClassification.from_pretrained(
                model_id,
                device_map=self.device
            )
            model.eval()
            
        return model, processor


    def __call__(
        self,
        image_pil,
        use_label_mapper=True
    ):
        if self.model_name in self.VIT_BASED_CLASSIFIERS:
            label_mapper = {
                "nsfw" : "NSFW",
                "sfw" : "SFW",
                "normal" : "SFW",
                # 'drawings' : "SFW", 
                # 'hentai' : "NSFW", 
                # 'neutral': "SFW", 
                # 'porn': "NSFW", 
                # 'sexy' : "SFW"
            }
            prediction = self.model(image_pil)

            if use_label_mapper:
                image_prediction_dict = {
                    label_mapper[pred_dict['label']] : round(pred_dict['score'], 3)
                    for pred_dict in prediction
                }
            else:
                image_prediction_dict = {
                    pred_dict['label'] : round(pred_dict['score'], 3)
                    for pred_dict in prediction
                }
        elif self.model_name == 'marqo':
            class_names = self.model.pretrained_cfg["label_names"]
            with torch.no_grad():
                output = self.model(
                    self.processor(image_pil).unsqueeze(0).to(self.device)
                ).softmax(dim=-1).cpu()[0]
        
            image_prediction_dict = {
                class_names[i] :  round(output[i].item(),3)
                for i in range(len(output))
            }
        else:
            inputs = self.processor(
                images=image_pil, 
                return_tensors="pt"
            ).to(self.device)
            labels =  ["SFW", "Questionable", "NSFW"]
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

            if use_label_mapper:
                label_mapper = {
                    "NSFW" : "NSFW",
                    "SFW" : "SFW",
                    "Questionable" : "NSFW",
                }
                
                image_prediction_dict = {
                    label_mapper[label] :  round(prob.item(), 3)
                    for label, prob in zip(labels, probabilities)
                }
            else:
                image_prediction_dict = {
                    label :  round(prob.item(), 3)
                    for label, prob in zip(labels, probabilities)
                }
            
        return image_prediction_dict
    

class NudeDetectorGPU(NudeDetector):
    def __init__(
        self, 
        model_path=None, 
        providers=None, 
        inference_resolution=320,
        device_id=0
    ):
        # print('Model path: ', model_path)
        # print('Inference resolution: ', inference_resolution)
        self.onnx_session = onnxruntime.InferenceSession(
            os.path.join(nudenet.__path__[0], "320n.onnx")
            if model_path is None
            else model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            provider_options=[{'device_id': device_id}, {}]
        )
        model_inputs = self.onnx_session.get_inputs()

        self.input_width = inference_resolution
        self.input_height = inference_resolution
        self.input_name = model_inputs[0].name


class NSFWDetector:
    def __init__(
        self,
        model_path=None,
        device_id=0
    ):
        self.device_id = device_id
        self.model_path = model_path
        model, processor = self.get_model_and_processor(
            model_path=self.model_path
        )
        self.model = model
        self.processor = processor

    
    def get_model_and_processor(
        self,
        model_path,
    ):
        model = None
        processor = None

        if model_path is None:
            inference_resolution = 320
        else:
            inference_resolution = 640

        model = NudeDetectorGPU(
            model_path=model_path, 
            inference_resolution=inference_resolution,
            device_id=self.device_id
        )
            
        return model, processor


    def __call__(
        self,
        images,
        batch_size=32
    ):
        predictions = self.model.detect_batch(
            images, 
            batch_size=batch_size
        )
        return predictions
    

def filter_predictions( 
    predictions_video, 
    selected_labels,
    score_threshold=0.3
):
    if not predictions_video:
        return predictions_video
        
    predictions_video_filtered = []

    for predictions_frame in predictions_video:
        predictions_frame_filtered = []
        
        for pred in predictions_frame:
            if not pred:
                continue
                
            box = pred['box']
            class_name = pred['class']
            score = pred['score']
    
            if class_name in selected_labels:
                if score>=score_threshold:
                    predictions_frame_filtered.append(pred)
                    
        predictions_video_filtered.append(predictions_frame_filtered)
        
    return predictions_video_filtered


def find_safe_video_segments(
    nsfw_scores, 
    threshold=0.5, 
    min_segment_length=10
):
    """
    Find the first segment of the video that doesn't contain NSFW content.
    
    Args:
        nsfw_scores: 1D numpy array of NSFW scores for each second
        threshold: NSFW score threshold to consider content as NSFW
        min_segment_length: Minimum length (in seconds) required for a clean segment
    """
    
    clean_indices = np.where(nsfw_scores < threshold)[0]
    
    if len(clean_indices) == 0:
        return None
    
    segments = []
    segment_start = clean_indices[0]
    
    for i in range(1, len(clean_indices)):
        if clean_indices[i] != clean_indices[i-1] + 1:
            segments.append((segment_start, clean_indices[i-1] + 1))
            segment_start = clean_indices[i]
    
    if len(clean_indices) > 0:
        segments.append((segment_start, clean_indices[-1] + 1))
    
    safe_segments = []
    for start, end in segments:
        if end - start >= min_segment_length:
            safe_segments.append([int(start), int(end)])

    return safe_segments
    

def find_safe_segments_combined(
    model_scores_dict, 
    threshold=0.5, 
    min_segment_length=5, 
    combination_method='mean'
):
    """
    Find the first clean segment using scores from multiple models.
    
    Args:
        model_scores_dict: Dictionary where keys are model names and values are 1D numpy arrays
                          of NSFW scores
        threshold: NSFW score threshold to consider content as NSFW
        min_segment_length: Minimum length (in seconds) required for a clean segment
        combination_method: How to combine scores ('max', 'mean', 'median')
    
    Returns:
        tuple: (start_time, end_time) of the first clean segment, or None if no valid segment found
    """
    if not model_scores_dict:
        return None
    
    model_scores_dict_classification = model_scores_dict['classification']
    model_arrays_classification = list(model_scores_dict_classification.values())
    array_lengths = [len(arr) for arr in model_arrays_classification]

    stacked_scores_detection = None
    if 'detection' in model_scores_dict:
        model_scores_dict_detection = model_scores_dict['detection']
        model_arrays_detection = list(model_scores_dict_detection.values())
        array_lengths += [len(model_arrays_detection[0])]
        stacked_scores_detection = np.vstack(model_arrays_detection)[0]

    if len(set(array_lengths)) > 1:
        raise ValueError("All model score arrays must have the same length")
    
    stacked_scores_classification = np.vstack(model_arrays_classification)
    
    if combination_method == 'max':
        combined_scores = np.max(stacked_scores_classification, axis=0)
    elif combination_method == 'mean':
        combined_scores = np.mean(stacked_scores_classification, axis=0)
    elif combination_method == 'median':
        combined_scores = np.median(stacked_scores_classification, axis=0) 
    else:
        raise ValueError("Invalid combination method. Choose 'max', 'mean', or 'median'")
    
    if stacked_scores_detection is not None:
        combined_scores = np.maximum(combined_scores, stacked_scores_detection)

    safe_segments = find_safe_video_segments(
        combined_scores, 
        threshold, 
        min_segment_length
    )

    return safe_segments, combined_scores


def process_subfolder(subfolder, save_dir, num_columns):
    subfolder_name = os.path.basename(subfolder)
    
    # Get all second images in the subfolder
    image_paths = sorted(glob.glob(os.path.join(subfolder, "second_*.jpg")), 
                        key=lambda x: int(os.path.basename(x).replace("second_", "").replace(".jpg", "")))
    
    if not image_paths:
        return f"No images found in {subfolder}"
        
    try:
        # Load the first image to get dimensions
        sample_img = Image.open(image_paths[0])
        img_width, img_height = sample_img.size
        
        # Calculate grid dimensions
        num_images = len(image_paths)
        num_rows = math.ceil(num_images / num_columns)
        
        # Create blank canvas for the collage
        collage_width = num_columns * img_width
        collage_height = num_rows * img_height
        collage = Image.new('RGB', (collage_width, collage_height))
        
        # Place images in the grid
        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path)
            row = i // num_columns
            col = i % num_columns
            x = col * img_width
            y = row * img_height
            collage.paste(img, (x, y))
        
        # Save the collage
        save_path = os.path.join(save_dir, f"{subfolder_name}.jpg")
        collage.save(save_path)
        return f"Created collage for {subfolder_name}"
    except Exception as e:
        return f"Error processing {subfolder}: {e}"


def worker_function(
    queue, 
    gpu_id,
    model_path_detector,
    save_dir_videos,
    save_dir_scores,
    save_dir_snapshots,
    save_dir_collages,
    frame_max_size,
    frame_interval_sec,
    n_sec_max,
    url_template, 
    secret,
    quality,
    score_threshold_nsfw,
    score_threshold_detector,
    min_segment_length,
    nsfw_offset,
    num_columns_collages,
    overwrite,
    counter, 
    lock
):
    video_downloader = VideoDownloader(
        url_template=url_template,
        secret=secret,
        quality=quality,
        save_dir=save_dir_videos,
        if_quality_not_exist_strategy='higher'
    )

    torch.cuda.set_device(gpu_id)

    video_frame_splitter_config = VideoFrameSplitterConfig(
        start_idx=0,
        frame_interval_sec=frame_interval_sec,
        frame_max_size=frame_max_size,
        n_sec_max=n_sec_max
    )
    video_frame_splitter = VideoFrameSplitter(
        config=video_frame_splitter_config
    )

    nsfw_classifiers_dict = {
        'falconsai' : NSFWClassifier(model_name='falconsai'),
        'marqo' : NSFWClassifier(model_name='marqo'),
        'lovetillion' : NSFWClassifier(model_name='lovetillion'),
    }

    for classifier_name in nsfw_classifiers_dict.keys():
        classifier_scores_save_dir = save_dir_scores / classifier_name
        classifier_scores_save_dir.mkdir(exist_ok=True)

    save_dir_scores_detector = save_dir_scores / 'nudenet640m'
    save_dir_scores_detector.mkdir(exist_ok=True)
    nsfw_detector = NSFWDetector(
        model_path=model_path_detector,
        device_id=gpu_id
    )

    model_names_classification = [
        'falconsai',
        'lovetillion',
        'marqo'
    ]

    model_names_detection = [
        'nudenet640m',
    ]

    combination_method = 'mean'
    
    while True:
        try:
            video_id = queue.get(timeout=1)  
            video_id_str = str(video_id)

            if video_id is None:  
                break
        except mp.queues.Empty:
            continue
        
        ############# Video downloader #############
        try:
            status_dict = video_downloader(video_id_str, verbose=False)
            video_path = Path(status_dict[video_id_str]['video_path'])

            if video_path is None:  
                print(f'Video is not exist: {video_id}')
                with lock: counter.value += 1
                continue

        except Exception as e:
            print(f'Video is not downloaded: {video_id}')
            print(str(e))
            with lock: counter.value += 1
            continue

        video_name = video_path.stem

        ############# Video splitter #############
        try:
            video_frames_data = video_frame_splitter(
                video_path, 
                verbose=False
            )
            frame_images = [
                frame_data.image
                for frame_data in video_frames_data.frames
            ]

            frame_images_pil = [
                Image.fromarray(image_np)
                for image_np in frame_images
            ]
        except Exception as e:
            print(f'Video can not be splitted: {video_id}')
            print(str(e))
            with lock: counter.value += 1
            continue
        
        ############# NSFW classifiers #############
        try:
            for classifier_name in nsfw_classifiers_dict.keys():
                classifier_scores_save_dir = save_dir_scores / classifier_name
                save_file_path_classifier = classifier_scores_save_dir / f'{video_name}.npy'
                if not save_file_path_classifier.is_file() or overwrite:
                    # print(f'Video {video_id_str}: classifier {classifier_name}')
                    nsfw_scores = []
                    for image_pil in frame_images_pil:    
                        classifier_model = nsfw_classifiers_dict[classifier_name]
                        image_prediction_dict = classifier_model(image_pil)
                        nsfw_scores.append(image_prediction_dict['NSFW'])
                    classifier_scores = np.array(nsfw_scores)
                    np.save(save_file_path_classifier, classifier_scores)
        except Exception as e:
            print(f'Something wrong with classifiers: {video_id}')
            print(str(e))
            with lock: counter.value += 1
            continue
        
        ############# Nudity detector #############
        try:
            save_file_path_detection_json = save_dir_scores_detector / f'{video_name}.json'
            if not save_file_path_detection_json.is_file() or overwrite:
                # print(f'Video {video_id_str}: detector nudenet640m')
                predictions_detector = nsfw_detector(
                    [frame_image.copy() for frame_image in frame_images], 
                    batch_size=32
                )

                with open(save_file_path_detection_json, "w") as f:
                    json.dump(predictions_detector, f, indent=4)
        except Exception as e:
            print(f'Something wrong with detectors: {video_id}')
            print(str(e))
            with lock: counter.value += 1
            continue

        ############# Convert detection results to classification #############
        try:
            save_file_path_detection_npy = save_dir_scores_detector / f'{video_name}.npy'
            if not save_file_path_detection_npy.is_file() or overwrite:
                with open(save_file_path_detection_json) as json_file:
                    predictions_detector = json.load(json_file)

                predictions_detector_filtered = filter_predictions( 
                    predictions_detector, 
                    selected_labels=NSFW_LABELS_DETECTOR,
                    score_threshold=score_threshold_detector
                )

                classification_predictions = []
                for predictions_frame_filtered in predictions_detector_filtered:
                    is_nsfw = int(len(predictions_frame_filtered)>0)
                    classification_predictions.append(is_nsfw)
                classification_predictions = np.array(classification_predictions)
                np.save(save_file_path_detection_npy, classification_predictions)
        except Exception as e:
            print(f'Something wrong on json to npy detection results convertion: {video_id}')
            print(str(e))
            with lock: counter.value += 1
            continue

        ############# SFW segments detection #############
        try:
            video_frame_scores_dict = {
                'classification' : {},
            }

            if len(model_names_detection)>0:
                video_frame_scores_dict['detection'] = {}

            for model_name_classification in model_names_classification:
                model_scores_dir = save_dir_scores / model_name_classification
                model_video_frame_scores_file_path = model_scores_dir / f'{video_name}.npy'
                model_video_frame_scores = np.load(model_video_frame_scores_file_path)
                video_frame_scores_dict['classification'][model_name_classification] = model_video_frame_scores

            for model_name_detection in model_names_detection:
                model_scores_dir = save_dir_scores / model_name_detection
                model_video_frame_scores_file_path = model_scores_dir / f'{video_name}.npy'
                model_video_frame_scores = np.load(model_video_frame_scores_file_path)
                video_frame_scores_dict['detection'][model_name_detection] = model_video_frame_scores

            # print('Video id: ', video_id_str)
            # for k, v in video_frame_scores_dict['classification'].items():
            #     print(k, ':', v.shape)
            # for k, v in video_frame_scores_dict['detection'].items():
            #     print(k, ':', v.shape)

            sfw_segments, combined_scores = find_safe_segments_combined(
                model_scores_dict=video_frame_scores_dict, 
                threshold=score_threshold_nsfw, 
                min_segment_length=min_segment_length, 
                combination_method=combination_method
            )

            video_results_save_dir = save_dir_snapshots / video_name
            video_results_save_dir.mkdir(exist_ok=True)

            safe_segments_dict = {
                'safe_segments' : sfw_segments
            }

            with open(video_results_save_dir / 'metadata.json', "w") as f:
                json.dump(safe_segments_dict, f, indent=4)

            if sfw_segments:
                sfw_segment = sfw_segments[0]
                if sfw_segment[0] == 0:
                    frame_seconds_to_read = min(sfw_segment[1] + nsfw_offset, len(combined_scores))
                    selected_seconds = list(range(frame_seconds_to_read))
                    selected_second_scores = combined_scores[:len(selected_seconds)]
                    
                    frames_vis = []
                    for frame_idx in range(frame_seconds_to_read):
                        frame_image = frame_images[frame_idx].copy()
                        if frame_idx < sfw_segment[1]:
                            vis_text = f'SFW (nsfw score: {selected_second_scores[frame_idx]:.2f})'
                            vis_bg_color = (0, 255, 0)
                        else:
                            vis_text = f'NSFW (nsfw score: {selected_second_scores[frame_idx]:.2f})'
                            vis_bg_color = (255, 0, 0)
                            
                        frame_vis = add_text_to_image(
                            frame_image, 
                            vis_text, 
                            bg_color=vis_bg_color, 
                            font_scale=0.5
                        )
                        frames_vis.append(frame_vis)
                        vis_image_save_path = video_results_save_dir / f'second_{frame_idx}.jpg'
                        frame_vis = cv2.cvtColor(frame_vis.copy(), cv2.COLOR_RGB2BGR)
                        cv2.imwrite(vis_image_save_path, frame_vis)

                    process_subfolder(
                        video_results_save_dir, 
                        save_dir_collages, 
                        num_columns_collages
                    )
        except Exception as e:
            print(f'Something wrong in sfw segments detection: {video_id}')
            print(str(e))
            with lock: counter.value += 1
            continue

        with lock: counter.value += 1


@click.command()
@click.option(
    "--dataset_file_path", 
    type=str, 
    required=True, 
)
@click.option(
    "--model_path_detector",
    default='./embeddings',
    type=str,
    required=True,
)
@click.option(
    "--save_dir_videos", 
    type=str, 
    default='./'
)
@click.option(
    "--save_dir_scores",
    default='./embeddings',
    type=str,
    required=True,
)
@click.option(
    "--save_dir_snapshots",
    default='./embeddings',
    type=str,
    required=True,
)
@click.option(
    "--save_dir_collages",
    default='./embeddings',
    type=str,
    required=True,
)
@click.option(
    "--frame_max_size", 
    type=int, 
    default=512
)
@click.option(
    "--frame_interval_sec", 
    type=int, 
    default=1
)
@click.option(
    "--n_sec_max", 
    type=int, 
    default=None
)
@click.option(
    "--url_template", 
    type=str, 
    required=True,
)
@click.option(
    "--secret", 
    type=str, 
    required=True,
)
@click.option(
    "--quality", 
    type=str, 
    default='max'
)
@click.option(
    "--n_gpus", 
    type=int, 
    default=4
)
@click.option(
    "--n_workers_per_gpu", 
    type=int, 
    default=2
)
@click.option(
    "--n_files_max", 
    type=int, 
    default=-1
)
@click.option(
    "--score_threshold_nsfw", 
    type=float, 
    default=0.7
)
@click.option(
    "--score_threshold_detector", 
    type=float, 
    default=0.25
)
@click.option(
    "--min_segment_length", 
    type=int, 
    default=10
)
@click.option(
    "--nsfw_offset", 
    type=int, 
    default=5
)
@click.option(
    '--num_columns_collages', 
    required=True, 
    type=int, 
    default=10,
    help='Number of columns in the collage grid'
)
@click.option(
    "--overwrite", 
    type=bool, 
    default=False
)
def process_videos(
    dataset_file_path,
    model_path_detector,
    save_dir_videos,
    save_dir_scores,
    save_dir_snapshots,
    save_dir_collages,
    frame_max_size,
    frame_interval_sec,
    n_sec_max,
    url_template,
    secret,
    quality,
    n_gpus,
    n_workers_per_gpu,
    n_files_max,
    score_threshold_nsfw,
    score_threshold_detector,
    min_segment_length,
    nsfw_offset,
    num_columns_collages,
    overwrite
):
    mp.set_start_method('spawn')

    dataset_file_path = Path(dataset_file_path)
    
    dataset_df = pd.read_csv(
        dataset_file_path, 
    )
    dataset_df['rating_total'] = dataset_df['rating_us'] + dataset_df['rating_eu']
    dataset_df = dataset_df[dataset_df['rating_total']>1000]
    dataset_df = dataset_df.sort_values('rating_total', ascending=False)
    video_ids = dataset_df.id.unique()

    if n_files_max>0:
        n_files_max = min(n_files_max, len(video_ids))
        video_ids = video_ids[:n_files_max]

    save_dir_videos = Path(save_dir_videos)
    save_dir_videos.mkdir(exist_ok=True)

    save_dir_scores = Path(save_dir_scores)
    save_dir_scores.mkdir(exist_ok=True)

    save_dir_snapshots = Path(save_dir_snapshots)
    save_dir_snapshots.mkdir(exist_ok=True)

    save_dir_collages = Path(save_dir_collages)
    save_dir_collages.mkdir(exist_ok=True)

    n_total_workers = n_gpus * n_workers_per_gpu
    task_queue = mp.Queue()
    counter = mp.Value('i', 0) 
    lock = mp.Lock()
    overwrite = bool(overwrite)

    print('N gpus: ', n_gpus)
    print('N workers per gpu : ', n_workers_per_gpu)
    print('Total N workers   : ', n_total_workers)
    print('Override predictions: ', overwrite)

    workers = []
    for gpu_id in range(n_gpus):
        for _ in range(n_workers_per_gpu):
            p = mp.Process(
                target=worker_function,
                args=(
                    task_queue, 
                    gpu_id,
                    model_path_detector,
                    save_dir_videos,
                    save_dir_scores,
                    save_dir_snapshots,
                    save_dir_collages,
                    frame_max_size,
                    frame_interval_sec,
                    n_sec_max,
                    url_template, 
                    secret,
                    quality,
                    score_threshold_nsfw,
                    score_threshold_detector,
                    min_segment_length,
                    nsfw_offset,
                    num_columns_collages,
                    overwrite,
                    counter, 
                    lock
                )
            )
            p.start()
            workers.append(p)

    for video_id in tqdm(video_ids, desc="Queueing videos"):
        task_queue.put(video_id)

    for _ in range(n_total_workers):
        task_queue.put(None)

    total_tasks = len(video_ids)

    pbar = tqdm(
        total=total_tasks, 
        desc="Downloading videos"
    )
    while counter.value < total_tasks:
        with lock:
            pbar.n = counter.value  
        pbar.refresh()
        time.sleep(1)

    pbar.n = total_tasks  
    pbar.close()

    for p in workers:
        p.join()

    print("All videos downloaded.")


if __name__ == '__main__':
    process_videos()
    
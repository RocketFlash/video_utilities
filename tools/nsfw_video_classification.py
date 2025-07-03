import click
import torch
import time
import timm
import numpy as np
from PIL import Image
import multiprocessing as mp
from transformers import pipeline
from transformers import (
    AutoProcessor, 
    FocalNetForImageClassification
)
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from video_utilities import (
    VideoFrameSplitter,
    VideoFrameSplitterConfig,
)


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


def worker_function(
    queue, 
    gpu_id, 
    frame_max_size,
    frame_interval_sec,
    n_sec_max,
    scores_save_dir, 
    counter, 
    lock
):
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
        # 'adamcodd' : NSFWClassifier(model_name='adamcodd'),
        'falconsai' : NSFWClassifier(model_name='falconsai'),
        # 'perry' : NSFWClassifier(model_name='perry'),
        'marqo' : NSFWClassifier(model_name='marqo'),
        'lovetillion' : NSFWClassifier(model_name='lovetillion'),
    }

    for classifier_name in nsfw_classifiers_dict.keys():
        classifier_scores_save_dir = scores_save_dir / classifier_name
        classifier_scores_save_dir.mkdir(exist_ok=True)
    
    while True:
        video_path = queue.get(timeout=1)  
        try:
            if video_path is None:  
                break
            video_name = video_path.stem
            video_frames_data = video_frame_splitter(video_path, verbose=False)
            frame_images = [
                frame_data.image
                for frame_data in video_frames_data.frames
            ]

            nsfw_scores = {
                classifier_name : []
                for classifier_name in nsfw_classifiers_dict.keys()
            }
            
            for frame_image in frame_images:
                image_np = frame_image
                image_pil = Image.fromarray(image_np)
                
                for classifier_name, classifier_model in nsfw_classifiers_dict.items():
                    image_prediction_dict = classifier_model(image_pil)
                    nsfw_scores[classifier_name].append(image_prediction_dict['NSFW'])

            for classifier_name in nsfw_classifiers_dict.keys():
                classifier_scores_save_dir = scores_save_dir / classifier_name
                save_file_path = classifier_scores_save_dir / f'{video_name}.npy'
                classifier_scores = np.array(nsfw_scores[classifier_name])
                np.save(save_file_path, classifier_scores)

            with lock:
                counter.value += 1
        except:
            print(f'Something wrong with {video_path}')
            continue


@click.command()
@click.option(
    "--dataset_dir", 
    type=str, 
    required=True, 
)
@click.option(
    "--scores_save_dir",
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
    "--n_gpus", 
    type=int, 
    default=4
)
@click.option(
    "--n_workers_per_gpu", 
    type=int, 
    default=1
)
def generate_nsfw_scores(
    dataset_dir,
    scores_save_dir,
    frame_max_size,
    frame_interval_sec,
    n_sec_max,
    n_gpus,
    n_workers_per_gpu
):
    mp.set_start_method('spawn')
    
    dataset_dir = Path(dataset_dir)
    video_paths = list(dataset_dir.glob('*.mp4'))
    
    scores_save_dir = Path(scores_save_dir)
    scores_save_dir.mkdir(exist_ok=True)

    n_total_workers = n_gpus * n_workers_per_gpu
    task_queue = mp.Queue()
    counter = mp.Value('i', 0) 
    lock = mp.Lock()

    print('N gpus: ', n_gpus)
    print('N workers per gpu : ', n_workers_per_gpu)
    print('Total N workers   : ', n_total_workers)

    workers = []
    for gpu_id in range(n_gpus):
        for _ in range(n_workers_per_gpu):
            p = mp.Process(
                target=worker_function,
                args=(
                    task_queue, 
                    gpu_id, 
                    frame_max_size,
                    frame_interval_sec,
                    n_sec_max,
                    scores_save_dir,
                    counter, 
                    lock
                )
            )
            p.start()
            workers.append(p)

    for video_path in tqdm(video_paths, desc="Queueing videos"):
        task_queue.put(video_path)

    for _ in range(n_gpus):
        task_queue.put(None)

    total_tasks = len(video_paths)

    pbar = tqdm(total=total_tasks, desc="Processing tasks")
    while counter.value < total_tasks:
        with lock:
            pbar.n = counter.value  
        pbar.refresh()
        time.sleep(1)

    pbar.n = total_tasks  
    pbar.close()

    for p in workers:
        p.join()

    print("All videos processed.")


if __name__ == '__main__':
    generate_nsfw_scores()
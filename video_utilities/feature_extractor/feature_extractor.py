import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
from .config import FeatureExtractorConfig
from transformers import AutoModel, AutoProcessor


class FeatureExtractor:
    def __init__(
        self,
        config=None,
    ):
        if config is None:
            config = self.get_default_config()

        self.config = config
        self.set_params_from_config(config)

        model, processor = self.get_model_and_processor(
            model_name=config.model_name
        )
        self.model = model
        self.processor = processor


    def get_default_config(self):
        return FeatureExtractorConfig()


    def set_params_from_config(
        self, 
        config: FeatureExtractorConfig
    ):
        for key, value in vars(config).items():
            setattr(self, key, value)


    def get_model_and_processor(
        self,
        model_name: str,
    ):
        model = AutoModel.from_pretrained(
            model_name, 
            device_map=self.device,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.dtype
        ).eval()
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor
    

    def generate_text_embeddings(
        self,
        texts,
    ):
        with torch.no_grad():
            text_inputs = self.processor(
                text=texts,
                images=None,
                return_tensors="pt",
                **self.processor_params
            ).to(self.device)
        
            with torch.autocast(
                device_type=self.device, 
                dtype=self.dtype
            ):
                text_outputs = self.model.get_text_features(**text_inputs)
                if self.normalize_vectors:
                    text_embeddings = text_outputs / text_outputs.norm(p=2, dim=-1, keepdim=True)
                text_embeddings = text_embeddings.detach().cpu()
        return text_embeddings


    def generate_image_embeddings(
        self,
        images,
    ):
        if not isinstance(images, list):
            images = [images]

        if isinstance(images[0], np.ndarray):
            images = [Image.fromarray(image) for image in images]

        with torch.no_grad():
            image_inputs = self.processor(
                images=images,
                text=None,
                return_tensors="pt",
            ).to(self.device)
            
            with torch.autocast(device_type=self.device, dtype=self.dtype):
                image_embeddings = self.model.get_image_features(**image_inputs)
                if self.normalize_vectors:
                    image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
                image_embeddings = image_embeddings.detach().cpu()

        return image_embeddings


    def calculate_similarity_scores(
        self,
        text_embeddings,
        image_embeddings,
    ):
        logits_per_text = torch.matmul(
            text_embeddings, 
            image_embeddings.t().to(text_embeddings.device)
        )
        
        logit_scale, logit_bias = self.model.logit_scale.to(text_embeddings.device), self.model.logit_bias.to(text_embeddings.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias
        
        logits_per_image = logits_per_text.t()
        scores_frames = torch.sigmoid(logits_per_image).detach().cpu().numpy()
        return scores_frames


    def calculate_label_scores(
        self,
        text_embeddings,
        image_embeddings,
        candidate_labels,
        normalize=False
    ):
        label_scores = {}

        scores_frames = self.calculate_similarity_scores(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
        )

        for scores_frame in scores_frames:
            if normalize:
                scores_sum = np.sum(scores_frame)
                scores = [score/scores_sum for score in scores_frame]
            else:
                scores = scores_frame
            
            for label, score in zip(candidate_labels, scores):
                if label not in label_scores:
                    label_scores[label] = [score]
                else:
                    label_scores[label].append(score)
        
        for k, v in label_scores.items():
            label_scores[k] = np.array(v)
            
        return label_scores


    def aggregate_embeddings_with_overlap(
        self,
        embeddings, 
        chunk_size, 
        overlap,
        mode='mean'
    ):
        stride = chunk_size - overlap
        n_frames = embeddings.shape[0]
        
        processed_chunks = []
        
        for start in range(0, n_frames - overlap, stride):
            end = min(start + chunk_size, n_frames)
            chunk = embeddings[start:end]
            
            if mode=='mean':    
                mean_vector = chunk.mean(dim=0)
            
            normalized_mean = mean_vector / mean_vector.norm(p=2, dim=-1, keepdim=True)        
            processed_chunks.append(normalized_mean)
        
        processed_embeddings = torch.stack(processed_chunks)
        
        return processed_embeddings


    def generate_image_embeddings_batches(
        self, 
        images,
        verbose: bool = False
    ):
        frame_embeddings = []
        bar = range(0, len(images), self.batch_size)

        if verbose:
            bar = tqdm(bar)

        for i in bar:
            batch_images = images[i:i+self.batch_size]
            embeddings_batch = self.generate_image_embeddings(batch_images)
            frame_embeddings.append(embeddings_batch)

        if verbose:
            bar.close()
        
        frame_embeddings = torch.vstack(frame_embeddings)
        
        return frame_embeddings


    def __call__(
        self, 
        images,
        verbose: bool = False,
        np_format: bool = True
    ):
        embeddings = self.generate_image_embeddings_batches(
            images,
            verbose=verbose
        )
        if np_format:
            embeddings = embeddings.numpy()

        return embeddings

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer

from datasets import load_dataset

from PIL import Image
import requests
from io import BytesIO
import logging
import random
from collections import deque

from config import *
from tokens import HF_TOKEN

class LaionDataset(Dataset):
    def __init__(
        self,
        config = DATASET_CONFIG,
        token = HF_TOKEN,
        split = "train",
        is_validation = False,
    ):
        self.config = config
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.token = token
        self.split = split
        self.timeout = config['timeout']
        self.max_attempts = config['max_attempts']
        self.shuffle = config['shuffle']
        self.shuffle_buffer_size = config['shuffle_buffer_size']

        self.min_text_length = config['min_text_length']
        self.max_text_length = config['max_text_length']

        self.image_size = config['image_size']
        self.min_image_size = config['min_image_size']
        self.max_image_size = config['max_image_size']

        self.random_crop_scale = config['random_crop_scale']
        self.random_crop_ratio = config['random_crop_ratio']
        self.color_jitter_prob = config['color_jitter_prob']
        self.color_jitter_params = config['color_jitter_params']
        self.grayscale_prob = config['grayscale_prob']
        self.gaussian_blur_prob = config['gaussian_blur_prob']
        self.gaussian_blur_kernel = config['gaussian_blur_kernel']
        self.gaussian_blur_sigma = config['gaussian_blur_sigma']
        self.horizontal_flip_prob = config['horizontal_flip_prob']

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        self.transform = self._setup_transforms()

        self.shuffle_buffer = deque(maxlen=self.shuffle_buffer_size)
        self.dataset = self.load_dataset()
        self.iterator = iter(self.dataset)

        self.is_validation = is_validation
        self.validation_buffer = []
        self.validation_size = config['validation_size']
        self.validation_frequency = config['validation_frequency']

    def __len__(self):
        if self.is_validation:
            return self.validation_size
        return int(2e9)
    
    def _fill_shuffle_buffer(self):
        while len(self.shuffle_buffer) < self.shuffle_buffer_size:
            try:
                sample = next(self.iterator)
                result = self._process_sample(sample)
                if result is not None:
                    self.shuffle_buffer.append(result)
            except StopIteration:
                self.iterator = iter(self.dataset)
                if len(self.shuffle_buffer) == 0:
                    continue
                break
            except Exception as e:
                self.logger.error(f"error shuffling buffer: {e}")
                continue
    
    def _fill_validation_buffer(self):
        while len(self.validation_buffer) < self.validation_size:
            try:
                sample = next(self.iterator)
                result = self._process_sample(sample)
                if result is not None:
                    self.validation_buffer.append(result)
            except StopIteration:
                self.iterator = iter(self.dataset)
            except Exception as e:
                self.logger.error(f"Error filling validation buffer: {e}")

    def __getitem__(self, _):
        if self.is_validation:
            if len(self.validation_buffer) < self.validation_size:
                self._fill_validation_buffer()
            return self.validation_buffer[idx % len(self.validation_buffer)]

        if not self.shuffle:
            for _ in range(self.max_attempts):
                try:
                    # Get next sample from iterator
                    sample = next(self.iterator)
                    result = self._process_sample(sample)
                    if result is not None:
                        return result
                except StopIteration:
                    # If iterator is exhausted, create new one
                    self.iterator = iter(self.dataset)
                    continue
                except Exception as e:
                    self.logger.debug(f"Error getting sample: {e}")
                    continue
            
            raise RuntimeError(f"Failed to get valid sample after {self.max_attempts} attempts")
        
        for _ in range(self.max_attempts):
            try:
                if len(self.shuffle_buffer) == 0:
                    self._fill_shuffle_buffer()
                
                if len(self.shuffle_buffer) == 0:
                    raise RuntimeError("No valid samples in dataset")
                
                idx = random.randint(0, len(self.shuffle_buffer) - 1)
                sample = self.shuffle_buffer[idx]
                
                self.shuffle_buffer.remove(sample)
                
                try:
                    new_sample = next(self.iterator)
                    result = self._process_sample(new_sample)
                    if result is not None:
                        self.shuffle_buffer.append(result)
                except StopIteration:
                    self.iterator = iter(self.dataset)
                except Exception as e:
                    self.logger.debug(f"Error adding new sample to buffer: {e}")
                
                return sample
                
            except Exception as e:
                self.logger.debug(f"Error in shuffle getitem: {e}")
                continue
    
    def _setup_transforms(self):
        """Setup image transforms based on training/validation mode"""
        if self.split == "train":
            return transforms.Compose([
                # Random resized crop
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=self.random_crop_scale,
                    ratio=self.random_crop_ratio,
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                
                # Color jitter with probability
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=self.color_jitter_params['brightness'],
                        contrast=self.color_jitter_params['contrast'],
                        saturation=self.color_jitter_params['saturation'],
                        hue=self.color_jitter_params['hue']
                    )
                ], p=self.color_jitter_prob),
                
                # Random horizontal flip
                transforms.RandomHorizontalFlip(p=self.horizontal_flip_prob),
                
                # Random grayscale
                transforms.RandomGrayscale(p=self.grayscale_prob),
                
                # Random gaussian blur
                transforms.RandomApply([
                    transforms.GaussianBlur(
                        kernel_size=self.gaussian_blur_kernel,
                        sigma=self.gaussian_blur_sigma
                    )
                ], p=self.gaussian_blur_prob),
                
                # Base transforms
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=DATASET_CONFIG['image_mean'],
                    std=DATASET_CONFIG['image_std']
                ),
            ])
        else:
            return transforms.Compose([
                # Validation transforms
                transforms.Resize(
                    self.image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=DATASET_CONFIG['image_mean'],
                    std=DATASET_CONFIG['image_std']
                ),
            ])

    def load_dataset(self):
        try:
            dataset = load_dataset(
                "laion/relaion2B-en-research-safe",
                streaming = True,
                token = self.token,
                split = self.split,
            )

            self.logger.info(f"dataset loaded successfully")
            return dataset

        except Exception as e:
            self.logger.error(f"error loading dataset: {e}")
            raise e

    def _validate_sample(self, sample):
        try:
            if not all (k in sample for k in ['url', 'caption']):
                return False
            
            if not sample['url'] or len(sample['url'].strip()) == 0:
                return False

            if not sample['caption'] or len(sample['caption'].strip()) < self.min_text_length:
                return False

            return True
        
        except Exception as e:
            self.logger.error(f"error validating sample: {e}")
            return False
        
    def _process_sample(self, sample):
        try:
            if not self._validate_sample(sample):
                return None
            
            image = self._process_image(sample['url'])
            if image is None:
                return None
            
            tokens = self._process_text(sample['caption'])
            if tokens is None:
                return None
            
            return {
                'image': image,
                'caption': sample['caption'],
                'input_ids': tokens['input_ids'],
                'attention_mask': tokens['attention_mask'],
            }
        
        except Exception as e:
            self.logger.error(f"error processing sample: {e}")
            return None
    
    def _process_text(self, text):
        try:
            tokens = self.tokenizer(
                text,
                padding = "max_length",
                max_length = self.max_text_length,
                truncation = True,
                return_tensors = "pt",
            )

            return {
                "input_ids": tokens.input_ids.squeeze(0),
                "attention_mask": tokens.attention_mask.squeeze(0),
            }
        
        except Exception as e:
            self.logger.error(f"error processing text: {e}")
            return None
    
    def _process_image(self, url):
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content))

            if image.mode != "RGB":
                image = image.convert("RGB")

            width, height = image.size

            if width < self.min_image_size or height < self.min_image_size:
                self.logger.debug(f"image size is too small: {width}x{height}")
                return None
            
            if width > self.max_image_size or height > self.max_image_size:
                self.logger.debug(f"image size is too large: {width}x{height}")
                return None
            
            image = self.transform(image)

            return image
        
        except Exception as e:
            self.logger.error(f"error loading image from {url}: {e}")
            return None
        
    def collate_fn(self, samples):
        samples = [s for s in samples if s is not None]
        
        if len(samples) == 0:
            raise RuntimeError("No valid samples found")
        
        batch = {
            'image': torch.stack([s['image'] for s in samples]),
            'input_ids': torch.stack([s['input_ids'] for s in samples]),
            'attention_mask': torch.stack([s['attention_mask'] for s in samples]),
        }

        return batch

    def get_dataloader(self):
        return DataLoader(
            self,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            collate_fn = self.collate_fn,
            pin_memory = True,
        )

    def _test_batch(self, num_batches=1):
        dataloader = self.get_dataloader()
        self.logger.info(f"Testing batch size: {self.batch_size}")

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            self.logger.info(f"Batch {i+1}")
            self.logger.info(f"Image shape: {batch['image'].shape}")
            self.logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
            self.logger.info(f"Attention mask shape: {batch['attention_mask'].shape}")

            if torch.isnan(batch['image']).any():
                self.logger.warning("Warning: Image contains NaN values")
            if torch.isinf(batch['image']).any():
                self.logger.warning("Warning: Image contains Inf values")
    
    def _test_load(self, num_samples=1):
        """
        Test the dataset loading and image processing
        Shows more detailed information about processed samples
        """
        
        self.logger.info(f"Testing dataset with {num_samples} samples")
        processed = 0
        attempts = 0
        max_attempts = num_samples * 10  # Allow for some failed samples
        
        for sample in self.dataset:
            if processed >= num_samples or attempts >= max_attempts:
                break
            
            attempts += 1
            result = self._process_sample(sample)
            
            if result is not None:
                processed += 1
                
                self.logger.info(f"\nSample {processed}:")
                self.logger.info(f"Caption: {result['caption'][:100]}...")  # Show first 100 chars
                self.logger.info(f"Image tensor shape: {result['image'].shape}")
                self.logger.info(f"Image value range: [{result['image'].min():.2f}, {result['image'].max():.2f}]")
                
                # Basic tensor checks
                if torch.isnan(result['image']).any():
                    self.logger.warning("Warning: Image contains NaN values")
                if torch.isinf(result['image']).any():
                    self.logger.warning("Warning: Image contains Inf values")
        
        success_rate = (processed / attempts) * 100 if attempts > 0 else 0
        self.logger.info(f"\nTesting complete:")
        self.logger.info(f"Processed {processed} valid samples out of {attempts} attempts")
        self.logger.info(f"Success rate: {success_rate:.1f}%")
    
    def _test_augmentations(self, num_samples=5, num_augmentations=3):
        """
        Test augmentations by applying them multiple times to the same image
        Saves the results for visual inspection
        """
        import matplotlib.pyplot as plt
        import torchvision.utils as vutils
        
        self.logger.info(f"Testing augmentations with {num_samples} samples")
        
        def denormalize(tensor):
            """Convert normalized image tensor back to [0,1] range"""
            mean = torch.tensor(DATASET_CONFIG['image_mean']).view(-1, 1, 1)
            std = torch.tensor(DATASET_CONFIG['image_std']).view(-1, 1, 1)
            return tensor * std + mean
        
        processed = 0
        attempts = 0
        
        while processed < num_samples and attempts < num_samples * 10:
            attempts += 1
            sample = next(self.iterator)
            
            try:
                # Get the original image
                response = requests.get(sample['url'], timeout=self.timeout)
                response.raise_for_status()
                original_image = Image.open(BytesIO(response.content)).convert('RGB')
                
                # Create a grid of augmented versions
                augmented_images = []
                for _ in range(num_augmentations):
                    aug_tensor = self.transform(original_image)
                    augmented_images.append(denormalize(aug_tensor))
                
                # Create grid
                grid = vutils.make_grid(
                    augmented_images, 
                    nrow=num_augmentations,
                    padding=2,
                    normalize=False
                )
                
                # Plot
                plt.figure(figsize=(15, 5))
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                plt.title(f"Sample {processed + 1}: Original + {num_augmentations-1} Augmentations")
                plt.axis('off')
                plt.savefig(f'test/augmentation_test_{processed+1}.png')
                plt.close()
                
                self.logger.info(f"\nSample {processed + 1}:")
                self.logger.info(f"Caption: {sample['caption'][:100]}...")
                self.logger.info("Augmented images saved as "
                            f"augmentation_test_{processed+1}.png")
                
                processed += 1
                
            except Exception as e:
                self.logger.error(f"Error processing sample: {e}")
                continue
        
        self.logger.info("\nAugmentation testing complete!")

if __name__ == "__main__":
    dataset = LaionDataset(shuffle_buffer_size=5)

    dataset._test_augmentations(num_samples=3, num_augmentations=4)
    dataset._test_load(num_samples=3)
    dataset._test_batch(num_batches=3)
 
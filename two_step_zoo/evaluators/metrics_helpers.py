from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as TF
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from pytorch_fid import fid_score
import inspect
from abc import ABC, abstractmethod
import sys


class FDHelper():
    def __init__(self, module, gt_loader, gen_samples, gen_batch_size, use_dinov2=True) -> None:
        self.module = module
        self.gt_loader = gt_loader
        self.gen_samples = gen_samples
        self.gen_batch_size = gen_batch_size
        self.use_dinov2 = use_dinov2
        if use_dinov2:
            self.fd_encoder = load_dinov2(module.device)
        else:
            self.fd_encoder = fid_score.InceptionV3().to(module.device)
        self.fd_encoder.eval()

    def gen_loader(self, **kwargs):
        # TODO: consider refactoring this into a `sample_loader` method in TwoStepDensityEstimator
        for i in range(0, self.gen_samples, self.gen_batch_size):
            if self.gen_samples - i < self.gen_batch_size:
                batch_size = self.gen_samples - i
            else:
                batch_size = self.gen_batch_size

            yield self.module.sample(batch_size, **kwargs), None, None

    def get_fd_features(self, im_loader=None, **kwargs):
        if im_loader:
            loader_len = len(self.gt_loader)
            loader_type = "ground truth"
        else:
            loader_len = self.gen_samples // self.gen_batch_size
            loader_type = "generated"
            im_loader = self.gen_loader(**kwargs)

        feats = []
        for batch, _, _ in tqdm(im_loader, desc=f"Getting {loader_type} features", leave=False, total=loader_len):
            # Convert grayscale to RGB
            if batch.ndim == 3:
                batch.unsqueeze_(1)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)

            if self.use_dinov2:
                # fd_encoder.transform was meant to be included in the dataloader and was copy-pasted from the DGM-eval
                # codebase, and thus only takes a single image (not a batch as input)
                batch = torch.cat([self.fd_encoder.transform(to_pil_image(x_)).unsqueeze(0) for x_ in batch])
            else:
                batch = batch / 255.

            batch = batch.to(self.module.device)

            with torch.no_grad():
                batch_feats = self.fd_encoder(batch)
                if not self.use_dinov2:
                    batch_feats = batch_feats[0]

            batch_feats = batch_feats.squeeze().cpu().numpy()
            feats.append(batch_feats)

        return np.concatenate(feats)

    def compute_fd_stats(self, im_loader=None, **kwargs):
        # Compute mean and covariance for generated and ground truth iterables
        feats = self.get_fd_features(im_loader, **kwargs)
        mu = np.mean(feats, axis=0)
        sigma = np.cov(feats, rowvar=False)

        return mu, sigma


def pil_resize(x, output_size):
    s1, s2 = output_size

    def resize_single_channel(x):
        img = Image.fromarray(x, mode='F')
        img = img.resize(output_size, resample=Image.BICUBIC)
        return np.asarray(img).clip(0, 255).reshape(s2, s1, 1)

    x = np.array(x.convert('RGB')).astype(np.float32)
    x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
    x = np.concatenate(x, axis=2).astype(np.float32)
    return to_tensor(x)/255


class Encoder(ABC, nn.Module):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        self.setup(*args, **kwargs)
        self.name = 'encoder'

    @abstractmethod
    def setup(self, *args, **kwargs):
        pass

    @abstractmethod
    def transform(self, x):
        """Converts a PIL Image to an input for the model"""
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


VALID_ARCHITECTURES = [
                        'vits14',
                        'vitb14',
                        'vitl14',
                        'vitg14',
                    ]


class DINOv2Encoder(Encoder):
    def setup(self, arch=None, clean_resize:bool=False):
        if arch is None:
            arch = 'vitl14'

        self.arch = arch

        arch_str = f'dinov2_{self.arch}'

        if self.arch not in VALID_ARCHITECTURES:
            sys.exit(f"arch={self.arch} is not a valid architecture. Choose from {VALID_ARCHITECTURES}")

        self.model = torch.hub.load('facebookresearch/dinov2', arch_str)
        self.clean_resize = clean_resize

    def transform(self, image):

        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])

        if self.clean_resize:
            image = pil_resize(image, (224, 224))
        else:
            image = TF.Compose([
                TF.Resize((224, 224), TF.InterpolationMode.BICUBIC),
                TF.ToTensor(),
            ])(image)

        return TF.Normalize(imagenet_mean, imagenet_std)(image)


def load_dinov2(device, **kwargs):
    """Load feature extractor"""

    model_cls = DINOv2Encoder

    # Get names of model_cls.setup arguments
    signature = inspect.signature(model_cls.setup)
    arguments = list(signature.parameters.keys())
    arguments = arguments[1:] # Omit `self` arg

    # Initialize model using the `arguments` that have been passed in the `kwargs` dict
    encoder = model_cls(**{arg: kwargs[arg] for arg in arguments if arg in kwargs})
    encoder.name = "dinov2"

    assert isinstance(encoder, Encoder), "Can only get representations with Encoder subclasses!"

    return encoder.to(device)

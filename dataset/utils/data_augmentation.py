import torch
import torch.nn.functional as F
from math import sin, cos, pi
import numbers
import random
from abc import ABC, abstractmethod

"""
    Data augmentation functions.

    There are some problems with torchvision data augmentation functions:
    1. they work only on PIL images, which means they cannot be applied to tensors with more than 3 channels,
       and they require a lot of conversion from Numpy -> PIL -> Tensor

    2. they do not provide access to the internal transformations (affine matrices) used, which prevent
       applying them for more complex tasks, such as transformation of an optic flow field (for which
       the inverse transformation must be known).

    For these reasons, we implement my own data augmentation functions
    (strongly inspired by torchvision transforms) that operate directly
    on Torch Tensor variables, and that allow to transform an optic flow field as well.
"""

class Augmentator(ABC):
    """Base class for all augmentations.
    """

    @abstractmethod
    def __call__(self, x, is_flow=False):
        pass


class Compose(Augmentator):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, is_flow=False):
        for t in self.transforms:
            x = t(x, is_flow)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Crop(Augmentator):
    """Center crop the tensor to a certain size.
    """

    def __init__(self, height, width, preserve_mosaicing_pattern=False):
        self.size = (int(height), int(width))

    def __call__(self, x, is_flow=False):
        """
            x: [T x C x H x W] Tensor to be cropped. | Dict of [T x C x H x W] Tensors to be cropped.
            is_flow: this parameter does not have any effect
        Returns:
            Tensor: Cropped tensor.
        """

        if isinstance(x, dict):
            _w = [x[key].shape[-1] for key in x]
            _h = [x[key].shape[-2] for key in x]

            # assert all tensors have the same size
            assert len(set(_w)) == 1
            assert len(set(_h)) == 1

            new_dict = {}
            for key in x:
                new_dict[key] = self.crop(x[key])
            return new_dict

        return self.crop(x)
        
    def crop(self, x):
        #Base Tensor case
        w, h = x.shape[-1], x.shape[-2]
        th, tw = self.size
        assert(th <= h)
        assert(tw <= w)
        return x[..., :th, :tw]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class CenterCrop(Augmentator):
    """Center crop the tensor to a certain size.
    """

    def __init__(self, height, width, preserve_mosaicing_pattern=False):
        self.size = (int(height), int(width))
        self.preserve_mosaicing_pattern = preserve_mosaicing_pattern

    def __call__(self, x, is_flow=False):
        """
            x: [T x C x H x W] Tensor to be cropped. | Dict of [T x C x H x W] Tensors to be cropped.
            is_flow: this parameter does not have any effect
        Returns:
            Tensor: Cropped tensor.
        """

        if isinstance(x, dict):
            _w = [x[key].shape[-1] for key in x]
            _h = [x[key].shape[-2] for key in x]

            # assert all tensors have the same size
            assert len(set(_w)) == 1
            assert len(set(_h)) == 1

            new_dict = {}
            for key in x:
                new_dict[key] = self.center_crop(x[key])
            return new_dict
        
        return self.center_crop(x)
    
    def center_crop(self, x):
        w, h = x.shape[-1], x.shape[-2]
        th, tw = self.size
        assert(th <= h), f"th: {th}, h: {h}"
        assert(tw <= w), f"tw: {tw}, w: {w}"
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        if self.preserve_mosaicing_pattern:
            # make sure that i and j are even, to preserve
            # the mosaicing pattern
            if i % 2 == 1:
                i = i + 1
            if j % 2 == 1:
                j = j + 1

        return x[..., i:i + th, j:j + tw]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomCrop(Augmentator):
    """Crop the tensor at a random location.
    """

    def __init__(self, height, width, preserve_mosaicing_pattern=False):
        self.size = (int(height), int(width))
        self.preserve_mosaicing_pattern = preserve_mosaicing_pattern

    @staticmethod
    def get_params(w, h, output_size):
        th, tw = output_size
        assert(th <= h)
        assert(tw <= w)
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return i, j, th, tw

    def __call__(self, x, is_flow=False):
        """
            x: [T x C x H x W] Tensor to be cropped. | Dict of [T x C x H x W] Tensors to be cropped.
            is_flow: this parameter does not have any effect
        Returns:
            Tensor: Cropped tensor.
        """

        if isinstance(x, dict):
            _w = [x[key].shape[-1] for key in x]
            _h = [x[key].shape[-2] for key in x]

            # assert all tensors have the same size
            assert len(set(_w)) == 1, f"Widths: {_w}"
            assert len(set(_h)) == 1, f"Heights: {_h}"

            w, h = _w[0], _h[0]

            i, j, h, w = self.get_params(w, h, self.size)
        else:
            w, h = x.shape[-1], x.shape[-2]
            i, j, h, w = self.get_params(w, h, self.size)

        if self.preserve_mosaicing_pattern:
            # make sure that i and j are even, to preserve the mosaicing pattern
            if i % 2 == 1:
                i = i + 1
            if j % 2 == 1:
                j = j + 1

        if isinstance(x, dict):
            new_dict = {}
            for key in x:
                new_dict[key] = self.crop(x[key], i, j, h, w)
            return new_dict

        return self.crop(x, i, j, h, w)
    
    def crop(self, x, i, j, h, w):
        return x[..., i:i + h, j:j + w]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomRotationFlip(Augmentator):
    """Rotate the image by angle. Include horizontal and vertical flips.
    """

    def __init__(self, degrees, p_hflip=0.5, p_vflip=0.5):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    @staticmethod
    def get_params(degrees, p_hflip, p_vflip):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])
        angle_rad = angle * pi / 180.0

        M_original_transformed = torch.FloatTensor([[cos(angle_rad), -sin(angle_rad), 0],
                                                    [sin(angle_rad), cos(angle_rad), 0],
                                                    [0, 0, 1]])

        if random.random() < p_hflip:
            M_original_transformed[:, 0] *= -1

        if random.random() < p_vflip:
            M_original_transformed[:, 1] *= -1

        M_transformed_original = torch.inverse(M_original_transformed)

        M_original_transformed = M_original_transformed[:2, :].unsqueeze(dim=0)  # 3 x 3 -> N x 2 x 3
        M_transformed_original = M_transformed_original[:2, :].unsqueeze(dim=0)

        return M_original_transformed, M_transformed_original

    def __call__(self, x, is_flow=False):
        """
            x: [T x C x H x W] Tensor to be rotated. | Dict of [T x C x H x W] Tensors to be rotated.
            is_flow: if True, x is an [T x 2 x H x W] displacement field, which will also be transformed
        Returns:
            Tensor: Rotated tensor.
        """

        #Skip assert... for now

        M_original_transformed, M_transformed_original = self.get_params(self.degrees, self.p_hflip, self.p_vflip)

        if isinstance(x, dict):
            new_dict = {}
            for key in x:
                _tmp_list = []
                for t in range(x[key].shape[0]):
                    affine_grid = F.affine_grid(M_original_transformed, x[key][t].unsqueeze(dim=0).shape, align_corners=False)
                    _tmp_list.append(self.rotate(x[key][t], affine_grid, M_transformed_original, is_flow))
                new_dict[key] = torch.stack(_tmp_list, dim=0)
            return new_dict
        
        _tmp_list = []
        for t in range(x.shape[0]):
            affine_grid = F.affine_grid(M_original_transformed, x.unsqueeze(dim=0).shape, align_corners=False)
            _tmp_list.append(self.rotate(x[t], affine_grid, M_transformed_original, is_flow))

        return torch.stack(_tmp_list, dim=0)

    def rotate(self, x, affine_grid, M_transformed_original, is_flow):
        transformed = F.grid_sample(x.unsqueeze(dim=0), affine_grid, align_corners=False)

        if is_flow:
            # Apply the same transformation to the flow field
            A00 = M_transformed_original[0, 0, 0]
            A01 = M_transformed_original[0, 0, 1]
            A10 = M_transformed_original[0, 1, 0]
            A11 = M_transformed_original[0, 1, 1]
            vx = transformed[:, 0, :, :].clone()
            vy = transformed[:, 1, :, :].clone()
            transformed[:, 0, :, :] = A00 * vx + A01 * vy
            transformed[:, 1, :, :] = A10 * vx + A11 * vy

        return transformed.squeeze(dim=0)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', p_flip={:.2f}'.format(self.p_hflip)
        format_string += ', p_vlip={:.2f}'.format(self.p_vflip)
        format_string += ')'
        return format_string
    
class RandomScale(Augmentator):
    """Scale the image by a factor.
    """

    def __init__(self, scale_factors):
        self.scale_factors = scale_factors

    @staticmethod
    def get_params(scale_factors):
        """Get parameters for ``scale`` for a random scale.
        Returns:
            sequence: params to be passed to ``scale`` for random scale.
        """
        scale = random.choice(scale_factors)
        return scale

    def __call__(self, x, is_flow=False):
        """
            x: [T x C x H x W] Tensor to be scaled. | Dict of [T x C x H x W] Tensors to be scaled.
            is_flow: if True, x is an [T x 2 x H x W] displacement field, which will also be transformed
        Returns:
            Tensor: Scaled tensor.
        """

        scale = self.get_params(self.scale_factors)

        if isinstance(x, dict):
            new_dict = {}
            for key in x:
                _tmp_list = []
                for t in range(x[key].shape[0]):
                    _tmp_list.append(self.scale(x[key][t], scale, is_flow))
                new_dict[key] = torch.stack(_tmp_list, dim=0)
            return new_dict
        
        _tmp_list = []
        for t in range(x.shape[0]):
            _tmp_list.append(self.scale(x[t], scale, is_flow))

        return torch.stack(_tmp_list, dim=0)

    def scale(self, x, scale, is_flow):
        if scale <= 1:
            transformed = F.interpolate(x.unsqueeze(dim=0), scale_factor=scale, mode='nearest', align_corners=False)
        else:
            transformed = F.interpolate(x.unsqueeze(dim=0), scale_factor=scale, mode='bicubic', align_corners=False)
        return transformed.squeeze(dim=0)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(scale_factor={0}'.format(self.scale_factor)
        format_string += ')'
        return format_string

def fetch_preprocessing(preprocessing_config):
    """
    Fetch the preprocessing pipeline from the configuration file.
    """
    preprocessing = []

    #Follow config list order

    for preproc_dict in preprocessing_config:
        if 'Crop' == preproc_dict['preprocessing_type']:
            _dict = {k:v for k,v in preproc_dict.items() if k != 'preprocessing_type'}
            preprocessing.append(Crop(**_dict))

        if 'CenterCrop' == preproc_dict['preprocessing_type']:
            _dict = {k:v for k,v in preproc_dict.items() if k != 'preprocessing_type'}
            preprocessing.append(CenterCrop(**_dict))

        if 'RandomCrop' == preproc_dict['preprocessing_type']:
            _dict = {k:v for k,v in preproc_dict.items() if k != 'preprocessing_type'}
            preprocessing.append(RandomCrop(**_dict))

        if 'RandomRotationFlip' == preproc_dict['preprocessing_type']:
            _dict = {k:v for k,v in preproc_dict.items() if k != 'preprocessing_type'}
            preprocessing.append(RandomRotationFlip(**_dict))
        
        if 'RandomScale' == preproc_dict['preprocessing_type']:
            _dict = {k:v for k,v in preproc_dict.items() if k != 'preprocessing_type'}
            preprocessing.append(RandomScale(**_dict))

    return Compose(preprocessing)
from torchvision.transforms.functional import InterpolationMode
# CONSTANT
MODEL_DICT = {
    'v1': 'runwayml/stable-diffusion-v1-5',
    'v2-base': 'stabilityai/stable-diffusion-2-base',
    'v2-large': 'stabilityai/stable-diffusion-2',
    'v2-1-base': 'stabilityai/stable-diffusion-2-1-base',
    'v2-1-large': 'stabilityai/stable-diffusion-2-1'
}
INTERPOLATIOND_DICT = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}
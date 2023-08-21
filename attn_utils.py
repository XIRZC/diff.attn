import argparse
import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode
from diffusers import StableDiffusionPipeline

import ptp_utils
from attn_const import MODEL_DICT, INTERPOLATIOND_DICT
from attn_lib import AttentionStore
    

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='v2-1-base', choices=list(MODEL_DICT.keys()), help='which stable diffusion model to leverage')
    parser.add_argument('--device', type=int, default=0, help='which gpu device to use')
    parser.add_argument('--seed', type=str, default=0, help='which seed to fix')
    parser.add_argument('--option', type=str, default="dis", help='generative or discriminative cross-attention show', choices=['gen', 'dis'])
    parser.add_argument('--prompt', type=str, default="A 'marc' passenger drains rides along railroad tracks.", help='which textual prompt to utlize')
    parser.add_argument('--img_size', type=int, default=512, help='Generated pixel space image size', choices=[256, 512])
    parser.add_argument('--image', type=str, default="./demo.png", help='image path to load for discrminative cross-attention show')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--vis', action='store_true', help='Whether visualize generated image and cross-attention map or not')
    parser.add_argument('--output', type=str, default="./attn_output/", help='Cross-attention and image output path')
    parser.add_argument('--num_steps', type=int, default=50, help='number of denoising timesteps for attention visualization')
    parser.add_argument('--cfg_scale', type=float, default=7.5, help='Classifier-free guidance scale')
    parser.add_argument('--low_resource', action='store_true', help='GPU memory greater than 12 or not (low-resource)')
    args, _ = parser.parse_known_args()
    return args


def center_crop_resize(img, interpolation=InterpolationMode.BILINEAR):
    transform = get_transform(interpolation=interpolation)
    return transform(img)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform


def aggregate_attention(attn_store, resolution, from_where, is_cross, select, prompt):

    out = []
    attention_maps = attn_store.get_average_attention()
    num_pixels = resolution ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompt), -1, resolution, resolution, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

    
def get_cross_attention(prompt, tokenizer, attn_store, resolution, from_where, select=0):

    tokens = tokenizer.encode(prompt[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attn_store, resolution, from_where, True, select, prompt)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    return np.stack(images, axis=0)


def get_self_attention_comp(attn_store, resolution, from_where, max_com=10, select=0):

    attention_maps = aggregate_attention(attn_store, resolution, from_where, False, select).numpy().reshape((resolution ** 2, resolution ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(resolution, resolution)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    return np.concatenate(images, axis=1)


def generative_attn_vis(args, prompt, sd_model, generator=None):

    attn_store = AttentionStore(args)
    gen_img, _ = ptp_utils.text2image_ldm_stable(sd_model, prompt, attn_store, num_inference_steps=args.num_steps,\
                                                  guidance_scale=args.cfg_scale, generator=generator, low_resource=args.low_resource)
    attn_img = get_cross_attention(prompt, sd_model.tokenizer, attn_store, resolution=16, from_where=("up", "down"))
    return gen_img, attn_img


def discriminative_attn_vis(args, image, prompt, sd_model, generator=None):

    attn_store = AttentionStore(args)
    interpolation = INTERPOLATIOND_DICT[args.interpolation]
    transform = get_transform(interpolation, args.img_size)
    image = transform(image).unsqueeze(0)
    latent = (sd_model.vae.encode(image.to(args.device)).latent_dist.mean) * 0.18215
    noise = torch.randn(
        (1, sd_model.unet.in_channels,  args.img_size // 8, args.img_size // 8),
        generator=generator,
    ).to(args.device)
    noised_latent = latent * (sd_model.scheduler.alphas_cumprod[args.num_steps] ** 0.5).view(-1, 1, 1, 1).to(args.device) + \
                        noise * ((1 - sd_model.scheduler.alphas_cumprod[args.num_steps]) ** 0.5).view(-1, 1, 1, 1).to(args.device)
    gen_img, _ = ptp_utils.text2image_ldm_stable(sd_model, prompt, attn_store, num_inference_steps=args.num_steps,\
                                     guidance_scale=args.cfg_scale, generator=generator,\
                                     latent=noised_latent, low_resource=args.low_resource)
    attn_img = get_cross_attention(prompt, sd_model.tokenizer, attn_store, resolution=16, from_where=("up", "down"))
    return gen_img, attn_img


def attn_vis(args, sd_model):
    
    generator = set_seed(args.seed)
    prompt = [args.prompt]

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    if args.option == 'gen':
        print(f"=> Running at generative cross-attention visualizing mode for prompt({args.prompt}) with seed({args.seed})")
        gen_img, gen_attn_img = generative_attn_vis(args, prompt, sd_model, generator=generator)
        print(f"=> Save gen_img_gen_mode and attn_vis_img_gen_mode into {output_dir}")
        ptp_utils.save_images(gen_img, args.vis, str(output_dir / f"gen_img_gen_mode.jpg"))
        ptp_utils.save_images(gen_attn_img, args.vis, str(output_dir / f"attn_vis_img_gen_mode.jpg"))
    elif args.option == 'dis':
        image = Image.open(f"{Path(args.image).resolve()}")
        print(f"=> Running at discriminative cross-attention visualizing mode for prompt({args.prompt}) and image({Path(args.image).resolve()}) with seed({args.seed})")
        gen_img, dis_attn_img = discriminative_attn_vis(args, image, prompt, sd_model, generator=generator)
        print(f"=> Save gen_img_dis and attn_vis_img_dis into {output_dir}")
        ptp_utils.save_images(gen_img, args.vis, str(output_dir / f"gen_img_dis_mode.jpg"))
        ptp_utils.save_images(dis_attn_img, args.vis, str(output_dir / f"attn_vis_img_dis_mode.jpg"))


if __name__ == '__main__':
    args = parse_args()
    sd_model = StableDiffusionPipeline.from_pretrained(MODEL_DICT[args.model]).to(f"cuda:{args.device}")
    attn_vis(args, sd_model)
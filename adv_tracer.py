import glob
import os
import pathlib
import argparse
from typing import Union, Tuple, List
from joblib import Parallel, delayed
from functools import partial

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder

from diffusers import ConsistencyModelPipeline, CMStochasticIterativeScheduler
from util import list_all_images, set_generator, normalize

class MultiDataset(Dataset):
    REPEAT_BY_APPEND: str = "REPEAT_BY_APPEND"
    REPEAT_BY_INPLACE: str = "REPEAT_BY_INPLACE"
    
    NOISE_MAP_SAME: str = "NOISE_MAP_SAME"
    NOISE_MAP_DIFF: str = "NOISE_MAP_DIFF"
    
    def __init__(self, src: Union[torch.Tensor, str, os.PathLike, pathlib.PurePath, List[Union[torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath]]], 
                 size: Union[int, Tuple[int, int], list[int]], 
                 repeat: int = 1, 
                 repeat_type: str=REPEAT_BY_INPLACE, 
                 noise_map_type: str=NOISE_MAP_SAME, 
                 vmin_out: float=-1.0, 
                 vmax_out: float=1.0, 
                 generator: Union[int, torch.Generator]=0, 
                 image_exts: list[str]=['png', 'jpg', 'jpeg', 'webp'], 
                 ext_case_sensitive: bool=False, 
                 njob: int=-1):
        if isinstance(src, list):
            self.__src__ = src
        elif isinstance(src, torch.Tensor):
            self.__src__ = torch.split(src, 1)
        elif isinstance(src, str) or isinstance(src, os.PathLike) or isinstance(src, pathlib.PurePath):
            self.__src__ = list_all_images(root=src, image_exts=image_exts, ext_case_sensitive=ext_case_sensitive)
        else:
            raise TypeError(f"Arguement src should not be {type(src)}, it should be Union[torch.Tensor, str, os.PathLike, pathlib.PurePath, List[Union[torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath]]]")
        
        self.__repeat__: int = repeat
        self.__repeat_type__: str = repeat_type
        self.__noise_map_type__: str = noise_map_type
        self.__generator__ = set_generator(generator=generator)
        self.__vmin_out__: float = vmin_out
        self.__vmax_out__: float = vmax_out
        self.__size__: Union[int, Tuple[int], List[int]] = size
        self.__njob__: int = njob
        self.__unique_src_n__: int = len(self)
        
        self.__build_src_noise__()
        
    def __repeat_data__(self, data: List):
        new_ls: List = []
        if self.__repeat_type__ == MultiDataset.REPEAT_BY_INPLACE:
            for e in data:
                new_ls = new_ls + [e] * self.__repeat__
            return new_ls
        elif self.__repeat_type__ == MultiDataset.REPEAT_BY_APPEND:
            return data * self.__repeat__
        else:
            raise ValueError(f"No such repeat type, {self.__repeat_type__}, should be {MultiDataset.REPEAT_BY_INPLACE} or {MultiDataset.REPEAT_BY_APPEND}")
    
    def __build_noise__(self):
        element_shape: List[int] = list(self.__loader__(data=self.__src__[0]).shape)
        if self.__noise_map_type__ == MultiDataset.NOISE_MAP_SAME:
            self.__noise__ = [x.squeeze(dim=0) for x in torch.randn(size=[self.__unique_src_n__] + element_shape, generator=self.__generator__).split(1)]
            self.__noise__ = self.__repeat_data__(data=self.__noise__)
        elif self.__noise_map_type__ == MultiDataset.NOISE_MAP_DIFF:
            self.__noise__ = [x.squeeze(dim=0) for x in torch.randn(size=[self.__unique_src_n__ * self.__repeat__] + element_shape, generator=self.__generator__).split(1)]
        else:
            raise ValueError(f"Arguement noise_map_type is {self.__noise_map_type__}, should be {MultiDataset.NOISE_MAP_SAME} or {MultiDataset.NOISE_MAP_DIFF}")
        
    def __build_src_noise__(self):
        self.__build_noise__()
        self.__src_idx__ = self.__repeat_data__(data=[i for i in range(self.__unique_src_n__)])
        self.__src__ = self.__repeat_data__(data=self.__src__)
    
    def __loader__(self, data: Union[torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath]):
        trans = [
            transforms.Resize(size=self.__size__),
            transforms.ConvertImageDtype(torch.float),
            # transforms.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=self.__vmin_out__, vmax_out=self.__vmax_out__, x=x)),
            transforms.Lambda(lambda x: (x - 0.5) * 2),
            # transforms.Normalize(mean=0, std=0.5),
        ]
        
        if isinstance(data, Image.Image):
            trans = [transforms.ToTensor()] + trans
        elif isinstance(data, str) or isinstance(data, os.PathLike) or isinstance(data, pathlib.PurePath):
            opening_image = transforms.Lambda(lambda x: Image.open(x).convert('RGB'))
            trans = [opening_image, transforms.ToTensor()] + trans
        else:
            raise TypeError(f"The arguement data should be torch.Tensor, Image.Image, str, os.PathLike, or pathlib.PurePath, not {type(image)}")
        
        return transforms.Compose(trans)(data)

    def __len__(self):
        return len(self.__src__)
    
    def to_tensor(self):
        res: List = list(self[0:len(self)])
        for i in range(len(res)):
            if isinstance(res[i][0], torch.Tensor):
                res[i] = torch.stack(res[i], dim=0)
            elif isinstance(res[i][0], int) or isinstance(res[i][0], float):
                res[i] = torch.tensor(res[i])
        return res

    def __getitem__(self, idx):
        src_ls: List = self.__src__[idx]
        if isinstance(src_ls, list):
            # return Parallel(n_jobs=self.__njob__)(delayed(self.__loader__)(e) for e in self.__src__[idx])
            return [self.__loader__(data=data) for data in src_ls], self.__noise__[idx], self.__src_idx__[idx]
            # return [self.__loader__(data=data) for data in src_ls]
        return self.__loader__(data=src_ls), self.__noise__[idx], self.__src_idx__[idx]
        # return self.__loader__(data=src_ls)
    
    def reverse(self, images: [torch.Tensor, List[torch.Tensor]]):
        trans = transforms.Compose([
            transforms.Lambda(lambda x: normalize(vmin_in=self.__vmin_out__, vmax_in=self.__vmax_out__, vmin_out=0, vmax_out=1, x=x)),
            # transforms.Normalize(mean=0, std=0.5),
        ])
        if isinstance(images, list):
            return [trans(img) for img in images]
        else:
            return trans(images)
    
    def save_images(self, images: [torch.Tensor, List[torch.Tensor]], file_names: Union[str, List[str]]):
        if isinstance(images, list) and isinstance(file_names, list):
            if len(images) != len(file_names):
                raise ValueError(f"The arguement images and file_names should be lists with equal length, not {len(images)} and {len(file_names)}.")
        rev_images = self.reverse(images=images)
        if isinstance(rev_images, list):
            for i, r_img in enumerate(rev_images):
                utils.save_image(r_img, fp=file_names[i])
        else:
            utils.save_image(rev_images, fp=file_names)
            
class Optimizer:
    OPTIM_ADAM: str = "ADAM"
    
    @staticmethod
    def get_tainable_param(trainable_param: torch.Tensor):
        return torch.nn.Parameter(trainable_param).requires_grad_(requires_grad=True)
    
    @staticmethod
    def optim_generator(name: str, lr: float, **kwargs):
        if name == Optimizer.OPTIM_ADAM:
            opt = partial(torch.optim.Adam, lr=lr, **kwargs)
        else:
            raise ValueError(f"")
        return opt
    
class LossFn:
    METRIC_L1: str = "L1"
    METRIC_L2: str = "L2"
    METRIC_FOURIER: str = "FOURIER"
    METRIC_SSIM: str = "SSIM"
    METRIC_LPIPS: str = "LPIPS"
    
    def get_loss_fn(metric: str, reduction: str='mean'):
        if metric == LossFn.METRIC_L1:
            criterion = torch.nn.L1Loss(reduction=reduction)
        elif metric == LossFn.METRIC_L2:
            criterion = torch.nn.MSELoss(reduction=reduction)
        # elif metric == LossFn.METRIC_SSIM:
        #     criterion = SSIMLoss()
        # elif metric == LossFn.METRIC_PSNR:
        #     criterion = psnr
        # elif metric == LossFn.METRIC_LPIPS:
        #     criterion = lpips_fn
        # elif metric == LossFn.METRIC_FOURIER:
        #     criterion = lpips_fn
        else:
            raise ValueError(f"Arguement metric doesn't support {metric}")
        return criterion

class ModelSched:
    MD_CLASS_CM: str = "MD_CM"
    
    MD_NAME_CD_L2_IMAGENET64: str = "CD_L2_IMAGENET64"
    MD_NAME_CD_LPIPS_IMAGENET64: str = "CD_LPIPS_IMAGENET64"
    MD_NAME_CT_IMAGENET64: str = "CT_IMAGENET64"
    
    MD_NAME_CD_L2_BEDROOM256: str = "CD_L2_BEDROOM256"
    MD_NAME_CD_LPIPS_BEDROOM256: str = "CD_LPIPS_BEDROOM256"
    MD_NAME_CT_BEDROOM256: str = "CT_BEDROOM256"
    
    MD_ID_CD_L2_IMAGENET64: str = "openai/diffusers-cd_imagenet64_l2"
    MD_ID_CD_LPIPS_IMAGENET64: str = "openai/diffusers-cd_imagenet64_lpips"
    MD_ID_CT_IMAGENET64: str = "openai/diffusers-ct_imagenet64"
    
    MD_ID_CD_L2_BEDROOM256: str = "openai/diffusers-cd_bedroom256_l2"
    MD_ID_CD_LPIPS_BEDROOM256: str = "openai/diffusers-cd_bedroom256_lpips"
    MD_ID_CT_BEDROOM256: str = "openai/diffusers-ct_bedroom256"
    
    @staticmethod
    def get_md_id(md_name: str):
        if md_name == ModelSched.MD_NAME_CD_L2_IMAGENET64:
            return ModelSched.MD_ID_CD_L2_IMAGENET64
        elif md_name == ModelSched.MD_NAME_CD_LPIPS_IMAGENET64:
            return ModelSched.MD_ID_CD_LPIPS_IMAGENET64
        elif md_name == ModelSched.MD_NAME_CT_IMAGENET64:
            return ModelSched.MD_ID_CT_IMAGENET64
        elif md_name == ModelSched.MD_NAME_CD_L2_BEDROOM256:
            return ModelSched.MD_ID_CD_L2_BEDROOM256
        elif md_name == ModelSched.MD_NAME_CD_LPIPS_BEDROOM256:
            return ModelSched.MD_ID_CD_LPIPS_BEDROOM256
        elif md_name == ModelSched.MD_NAME_CT_BEDROOM256:
            return ModelSched.MD_ID_CT_BEDROOM256
        else:
            raise ValueError(f"Arguement md_name, {md_name}, doesn't support ")
    
    @staticmethod
    def get_model_sched(model_type: str, model_id: str):
        vae = None
        # model, vae, noise_sched, get_pipeline, lora_layers = DiffuserModelSched.get_model_sched(ckpt=ep_model_path, clip_sample=config.clip, noise_sched_type=config.sched, sde_type=config.sde_type, rank=config.lora_rank, alpha=config.lora_alpha)
        pipe: DiffusionPipeline = ConsistencyModelPipeline.from_pretrained(ModelSched.get_md_id(model_id), torch_dtype=torch.float16)
        return pipe, pipe.unet.eval(), vae, pipe.scheduler
    
    @staticmethod
    def all_compile(*args):
        comps: List = []
        for comp in args:
            if comp is None:
                comps.append(comp)
            else:
                comps.append(torch.compile(comp, mode="reduce-overhead", fullgraph=True))
        return comps

    @staticmethod
    def all_to_device(*args, device: Union[str, torch.device]):
        comps: List = []
        for comp in args:
            if hasattr(comp, 'to'):
                comps.append(comp.to(device))
            else:
                comps.append(comp)
        return comps

def single_step_denoise(pipeline, latents: torch.Tensor, from_t: int=0):
    return (pipeline(batch_size=len(latents), num_inference_steps=1, class_labels=None, latents=latents, output_type='pt').images - 0.5) * 2
    # return (latents - epsilon) / 

def get_dataset(root: str, size: Union[int, Tuple[int], List[int]], vmin_out: int=-1, vmax_out: int=1):
    return MultiDataset(src=root, size=size, vmin_out=vmin_out, vmax_out=vmax_out)

def get_dataloader(dataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def batchify(data, batch_size: int):
    res: list = []
    for i in range(0, len(data), batch_size):  
        res.append(data[i:i + batch_size])
    return res

def sampling(pipeline, num: int, path: str, num_inference_steps: int=1, prompts: List=None, prompts_arg_name: str=None, batch_size: int=32, generator: torch.Generator=None):
    if prompts is None or len(prompts) == 0:
        used_prompts: List = [None for i in range(num)]
    else:
        used_prompts: List = [prompts[i % len(prompts)] for i in range(num)]
        
    generator = set_generator(generator=generator)
    
    os.makedirs(path, exist_ok=True)
    for batch_idx, prompt in enumerate(batchify(data=used_prompts, batch_size=batch_size)):
        batch_size: int = len(prompt)
        if prompts is None:
            prompt = None
            
        if prompts_arg_name is not None:
            images = pipeline(batch_size=batch_size, num_inference_steps=num_inference_steps, generator=generator, **{prompts_arg_name: prompt}).images
        else:
            if isinstance(prompt, int):
                images = pipeline(batch_size=batch_size, num_inference_steps=num_inference_steps, generator=generator, class_labels=prompt).images
            elif isinstance(prompt, str):
                images = pipeline(batch_size=batch_size, num_inference_steps=num_inference_steps, generator=generator, prompt=prompt).images
            else:
                images = pipeline(batch_size=batch_size, num_inference_steps=num_inference_steps, generator=generator, prompt=prompt).images
            
        for idx, image in enumerate(images):
            image.save(os.path.join(path, f"{batch_idx* batch_size + idx}.jpg"))

def tmp_ds():
    ds = get_dataset(root="fake_images/celeba_hq_256_ddpm", size=64)
    print(f"ds[0:5] Len: {len(ds[0:5])}, len(ds[0:5][0]): {len(ds[0:5][0])}")
    print(f"ds.to_tensor() Len: {len(ds.to_tensor())}, ds.to_tensor()[0].shape: {ds.to_tensor()[0].shape}, ds.to_tensor()[1].shape: {ds.to_tensor()[1].shape}, ds.to_tensor()[2].shape: {ds.to_tensor()[2].shape}")
    print(f"Max: {ds[0][0].max()}, Min: {ds[0][0].min()}")
    # utils.save_image(ds.reverse(ds[0]), fp='test.jpg')
    ds.save_images(ds[0][0], 'test.jpg')
    ds.save_images([ds[0:2][0][0], ds[0:2][1][0]], ['test.jpg', 'test1.jpg'])

def tmp_fn():
    model_type: str = ModelSched.MD_CLASS_CM
    model_id: str = ModelSched.MD_NAME_CD_LPIPS_IMAGENET64
    num_classes: int = 1000
    sample_num: int = 3000
    path: str = os.path.join("fake_images", model_id)
    num_inference_steps: int = 1
    prompts: List[int] = [i for i in range(num_classes)]
    # prompts: List[int] = None
    prompts_arg_name: str = 'class_labels'
    batch_size: int = 32
    device: Union[str, torch.device] = 'cuda:1'
    seed: int = 0
    
    g = torch.Generator(device=device).manual_seed(seed)
    
    pipe, unet, vae, scheduler = ModelSched.get_model_sched(model_type=model_type, model_id=model_id)
    pipe, unet, vae, scheduler = ModelSched.all_to_device(pipe, unet, vae, scheduler, device=device)
    pipe.unet, unet, vae = ModelSched.all_compile(pipe.unet, unet, vae)
    sampling(pipeline=pipe, num=sample_num, path=path, num_inference_steps=num_inference_steps, prompts=prompts, prompts_arg_name=prompts_arg_name, batch_size=batch_size, generator=g)
    
def tmp_train():
    ds_root: str = "fake_images/celeba_hq_256_ddpm"
    vmin_out: int = -1
    vmax_out: int = 1
    model_type: str = ModelSched.MD_CLASS_CM
    model_id: str = ModelSched.MD_NAME_CD_LPIPS_IMAGENET64
    loss_metric: str = LossFn.METRIC_L2
    lr: float = 0.001
    num_classes: int = 1000
    sample_num: int = 3000
    path: str = os.path.join("fake_images", model_id)
    num_inference_steps: int = 1
    prompts: List[int] = [i for i in range(num_classes)]
    # prompts: List[int] = None
    prompts_arg_name: str = 'class_labels'
    batch_size: int = 32
    device: Union[str, torch.device] = 'cuda:1'
    seed: int = 0
    max_iter: int = 1000
    
    g = torch.Generator(device=device).manual_seed(seed)
    
    pipe, unet, vae, scheduler = ModelSched.get_model_sched(model_type=model_type, model_id=model_id)
    loss_fn = LossFn.get_loss_fn(metric=loss_metric)
    pipe, unet, vae, scheduler, loss_fn = ModelSched.all_to_device(pipe, unet, vae, scheduler, loss_fn, device=device)
    pipe.unet, unet, vae = ModelSched.all_compile(pipe.unet, unet, vae)
    
    image_size: Union[int, List[int], Tuple[int]] = pipe.unet.config.sample_size
    
    ds = get_dataset(root=ds_root, size=image_size, vmin_out=vmin_out, vmax_out=vmax_out)
    dl = get_dataloader(dataset=ds, batch_size=batch_size)
    
    for batch_idx, batch in enumerate(dl):
        img, noise, img_id = batch
        img, noise = ModelSched.all_to_device(img, noise, device=device)
        noise = Optimizer.get_tainable_param(noise)
        optim = Optimizer.optim_generator(name=Optimizer.OPTIM_ADAM, lr=lr)([noise])
        
        print(f"Batch {batch_idx}")
        
        # progress_bar = tqdm(total=max_iter, disable=not accelerator.is_local_main_process)
        progress_bar = tqdm(total=max_iter)
        progress_bar.set_description(f"Batch {batch_idx}")
        for i in tqdm(range(max_iter)):
            loss = loss_fn(noise.float(), single_step_denoise(pipeline=pipe, latents=noise, from_t=0).float())
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            
            torch.cuda.empty_cache()
    
if __name__ == "__main__":
    # tmp_ds()
    # tmp_fn()
    tmp_train()
    
    
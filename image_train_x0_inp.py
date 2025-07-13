"""
Train a diffusion model on images.
"""

import argparse
from clearml import Task, Dataset
from scripts.guided_diffusion import dist_util, logger
from scripts.guided_diffusion.image_datasets import load_data
from scripts.guided_diffusion.resample import create_named_schedule_sampler
from scripts.guided_diffusion.script_util_x0 import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from scripts.guided_diffusion.train_util import TrainLoop
import torch.distributed as dist
import torch as th


def main():
    args = create_argparser().parse_args()
    print("\nARGS=", args,"\n")

    dist_util.setup_dist()
    logger.configure()

    
    task = Task.init(project_name='TP601375_DiffusionDenoiser', 
                 task_name='TP602603_ImageDenoiserTraining', 
                 output_uri='https://files.clearml.thefoundry.co.uk')
    task.upload_artifact('summaries', artifact_object='./clearml_summary') # Access to summary folder or .zip file
    task.set_packages('requirements.txt')
    #task.connect_configuration('./configs/config_train_generatore_size256_channels256.yaml')
    

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
  
    # Load pre-trained model
    upload_task_id_of_pretrained_model = '1c7287c02b4344f08e15f32858d6a582' # e.g., 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6'
    artifact_name_of_pretrained_model = 'pretrained_256x256_diffusion_uncond' # The name you gave in task.upload_artifact()
    logger.log(f"Attempting to retrieve pre-trained model artifact '{artifact_name_of_pretrained_model}' from task {upload_task_id_of_pretrained_model}...")
    # Get the task that uploaded the artifact
    upload_task = Task.get_task(task_id=upload_task_id_of_pretrained_model)
    # Get the artifact object
    pretrained_model_artifact = upload_task.artifacts[artifact_name_of_pretrained_model]
    logger.log("PRETRAINED MODEL ARTIFACT=", pretrained_model_artifact)
    # Download the artifact file to a temporary local path
    local_pretrained_model_path = pretrained_model_artifact.get_local_copy()
    model_state_dict = th.load(local_pretrained_model_path, map_location='cpu') # Load state dict
    model.load_state_dict(model_state_dict)
    logger.log("Pre-trained model loaded into architecture successfully!")
    
    model.to(dist_util.dev())
    logger.log("Model loaded to", dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    

    logger.log("creating data loaders...")
    
    guidance_dataset = Dataset.get(dataset_name = "Imagenet256_clean_for_TP602603")
    guidance_dataset_path = guidance_dataset.get_local_copy()
    guidance_dataset_meta = guidance_dataset.get_metadata()
    
    guidance_data = load_data(
        data_dir=args.guidance_images,   # For running locally
        #data_dir = guidance_dataset_path, # For running on clearml
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
 
    noisy_dataset = Dataset.get(dataset_name = "Imagenet256_noisy_fuji250under_for_TP602603")
    noisy_dataset_path = noisy_dataset.get_local_copy()
    noisy_dataset_meta = noisy_dataset.get_metadata()
    
    noisy_start_data = load_data(
        data_dir=args.noisy_start_images, # For running locally
        #data_dir = noisy_dataset_path,     # For running on clearml
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        noisy_start_data=noisy_start_data,
        guidance_data=guidance_data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()
    task.close()
    dist.destroy_process_group()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        # from GDP sample_x0_inp.py
        clip_denoised=True,
        num_samples=100,
        use_ddim=False,
        model_path="../scripts/models/256x256_diffusion_uncond.pt"

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)



    guidance_images = './scripts/imagenet_dataloader/imagenet256_clean.npz'
    noisy_start_images = './scripts/imagenet_dataloader/imagenet256_noisy_fuji250under.npz'
    #noisy_start_images = './scripts/imagenet_dataloader/imagenet256_noisy_mixed.npz'
    # add zhaoyang own's arguments
    parser.add_argument("--device", default=0, type=int, help='the cuda device to use to generate images')
    parser.add_argument("--global_rank", default=0, type=int, help='global rank of this process')
    parser.add_argument("--world_size", default=1, type=int, help='the total number of ranks')
    #parser.add_argument("--save_dir", default=save_dir, type=str, help='the directory to save the generated images')
    #parser.add_argument("--save_png_files", action='store_true', help='whether to save the generate images into individual png files')
    #parser.add_argument("--save_numpy_array", action='store_true', help='whether to save the generate images into a single numpy array')
    
    # these two arguments are only valid when not start from scratch
    parser.add_argument("--denoise_steps", default=25, type=int, help='number of denoise steps')
    #parser.add_argument("--dataset_path", default=None, type=str, help='path to the noisy images to start the diffusion steps. Could be an npz file or an image folder')
    parser.add_argument("--noisy_start_images", default=noisy_start_images, type=str, help='path to  the noisy images to start the diffusion steps. Could be an npz file or an image folder')
    # Turning off --use_img_for_guidance breaks the general_cond_fn.
    parser.add_argument("--use_img_for_guidance", action='store_true', help='whether to use a (low resolution) image for guidance. If true, we generate an image that is similar to the low resolution image')
    parser.add_argument("--img_guidance_scale", default=4000, type=float, help='guidance scale')
    parser.add_argument("--guidance_images", default=guidance_images, type=str, help='the directory or npz file to the guidance imgs. This folder should have the same structure as dataset_path, there should be a one to one mapping between images in them')
    parser.add_argument("--sample_noisy_x_lr", action='store_true', help='whether to first sample a noisy x_lr, then use it for guidance. ')
    parser.add_argument("--sample_noisy_x_lr_t_thred", default=1e8, type=int, help='only for t lower than sample_noisy_x_lr_t_thred, we add noise to lr')  
    parser.add_argument("--start_from_scratch", action='store_true', help='whether to generate images purely from scratch, not use gan or vae generated samples')
    parser.add_argument("--deg", default='inp25', type=str, help='the chosen of degradation model')
    # num_samples is defined elsewhere, num_samples is only valid when start_from_scratch and not use img as guidance
    # if use img as guidance, num_samples will be set to num of guidance images
    # parser.add_argument("--num_samples", type=int, default=50000, help='num of samples to generate, only valid when start_from_scratch is true')
    
    return parser



if __name__ == "__main__":
    main()

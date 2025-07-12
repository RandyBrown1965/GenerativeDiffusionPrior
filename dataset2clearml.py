from clearml import Dataset

# Create the new dataset. It should appear in the ClearML dashboard as "Uploading"
#dataset_dir_name = "imagenet256_clean"
#dataset_dir_name = "imagenet256_noisy_fuji250under"
dataset_dir_name = "imagenet256_noisy_mixed"

ds = Dataset.create(
    dataset_project='TP601375_DiffusionDenoiser',
    #dataset_name='Imagenet256_clean_for_TP602603',
    dataset_name = f'{dataset_dir_name}_for_TP602603',
    dataset_version='None')
    
# Specify files to add to the dataset. Presumably this can be called multiple times.
ds.add_files(
    #path='datasets/imagenet256_clean', # e.g. /mnt/qnap2/Projects/FaceTools/datasets/vfhq
    path=f'./scripts/imagenet_dataloader/{dataset_dir_name}', # e.g. /mnt/qnap2/Projects/FaceTools/datasets/vfhq
    wildcard='*.png',        # Or *.png or whatever
    local_base_folder=None,
    dataset_path=None,
    recursive=True
)

# This is the slow part - uploading the dataset will involve zipping up local files and transfering them
ds.upload(
    show_progress=True,
    verbose=True,
    output_url=None,
    compression=None
)

# Finalizing a dataset makes it ready for use. It's status should change on the dashboard
ds.finalize()
# Publishing makes the dataset read-only, which you might not want
# ds.publish()
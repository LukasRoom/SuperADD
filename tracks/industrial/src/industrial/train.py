from industrial.dataset import MVTecAD2Dataset
from industrial.model import SuperADD
import numpy as np
from torchvision import transforms
from industrial.paths import get_root_config, get_dataset_path, get_model_path


def main() -> None:

    config = get_root_config()

    dataset_path = get_dataset_path('mvtec_ad_2')

    for category in config['categories']:
        print(f'processing category {category}')

        # reproducible seed
        np.random.seed(42)

        train_data = MVTecAD2Dataset(dataset_path, category, 'train', transform=transforms.ToTensor())

        # access subsampled fraction of train data
        train_images = [train_data[i].image for i in range(0, len(train_data), config['train_fraction'])]

        # create model
        model = SuperADD(backbone=config['backbone'],
                         layers=config['layers'],
                         resize_factor=config['patch_size'] / 1024,
                         patch_size=config['patch_size'],
                         patch_overlap=config['patch_overlap'],
                         max_database_size=config['max_database_size'],
                         threshold_fraction=config['threshold_fraction'],
                         subsampling_iterations=config['subsampling_iterations'],
                         threshold_percentile=config['threshold_percentile'],
                         threshold_factor=config['threshold_factor'],
                         evaluation_downscale=config['evaluation_downscale'],
                         closing_radius=config['closing_radius'],
                         closing_angles=config['closing_angles'],
                         closing_lower_threshold=config['closing_lower_threshold'],
                         binary_erosion=config['binary_erosion'],
                         brightness_augmentation=config['brightness_augmentation'],
                         device='cuda')

        # train model on anomaly free images
        model.train(train_images)

        # store model to disk
        model_path = get_model_path(category)
        model.to_disk(model_path)

if __name__ == "__main__":
    main()

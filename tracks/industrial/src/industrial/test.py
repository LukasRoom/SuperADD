import os.path
import numpy as np
from industrial.model import SuperADD
from industrial.paths import (get_root_config, get_dataset_path, get_model_path, get_result_anomaly_images_path,
                              get_result_anomaly_images_thresholded_path, get_result_csv_path)
from industrial.dataset import MVTecAD2Dataset
from torchvision import transforms
from tqdm import tqdm
import sys
from sklearn.metrics import roc_auc_score, f1_score
import tifffile as tiff
import cv2
import pandas as pd


def main() -> None:
    config = get_root_config()

    dataset_path = get_dataset_path('mvtec_ad_2')

    for split in config['test_split']:

        print(f'\nprocessing split {split}')

        category_f1_scores = {}
        category_au_scores = {}

        for category in config['categories']:
            model = SuperADD.from_disk(get_model_path(category))

            test_data = MVTecAD2Dataset(dataset_path, category, split, transform=transforms.ToTensor())

            anomaly_maps = []
            binary_maps = []
            ground_truths = []

            for sample in tqdm(test_data, desc=f'processing {category} images', file=sys.stdout):

                basename = os.path.basename(sample.image_path).removesuffix('.png')

                if sample.label == 0 and not config['evaluate_good_images']:
                    continue

                anomaly_map, binary_result = model.predict(sample.image)

                anomaly_map = anomaly_map.astype(np.float16)

                if config['save_predictions']:
                    tiff.imwrite(get_result_anomaly_images_path(category, split) / f'{basename}.tiff', anomaly_map)
                    cv2.imwrite(get_result_anomaly_images_thresholded_path(category, split) / f'{basename}.png', binary_result)

                if sample.label == -1:
                    continue

                if sample.mask is None:
                    ground_truth = np.zeros_like(binary_result)
                else:
                    ground_truth = sample.mask.numpy().astype(np.uint8)[0]
                    ground_truth = cv2.resize(ground_truth, binary_result.shape[::-1], interpolation=cv2.INTER_LINEAR)

                anomaly_maps.append(anomaly_map)
                binary_maps.append(binary_result)
                ground_truths.append(ground_truth)

            if len(ground_truths) > 0:
                anomaly_maps = np.array(anomaly_maps).ravel()
                binary_maps = np.array(binary_maps).ravel() > 0
                ground_truths = np.array(ground_truths).ravel() > 0

                roc_auc = roc_auc_score(ground_truths, anomaly_maps, max_fpr=0.05)
                f1 = f1_score(ground_truths, binary_maps)

                print(f'category {category}: {roc_auc*100:.2f}% (AU), {f1*100:.2f}% (F1)')
                category_au_scores[category] = roc_auc
                category_f1_scores[category] = f1

        if len(category_f1_scores) > 0:
            print(f'\n---------- Results -----------')
            print(f'{split}')
            results_df = pd.DataFrame(
                zip(category_f1_scores.keys(), category_au_scores.values(), category_f1_scores.values()),
                columns=['Category', 'AU', 'F1'])
            results_df.loc[len(results_df)] = ['mean', results_df['AU'].mean(), results_df['F1'].mean()]
            results_df.to_csv(get_result_csv_path(f'scores_{split}'))
            print(results_df)


if __name__ == "__main__":
    main()

import argparse
import os

from scipy.spatial.distance import cosine
from tqdm import tqdm

from torchreid.utils import FeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_name")
    parser.add_argument("model_path")
    parser.add_argument("--query_data_root", help="Path to the query images")
    parser.add_argument("--query_annotation_path",
                        help="Path to the query annotations")
    parser.add_argument("--gallery_data_root",
                        help="Path to the gallery images")
    parser.add_argument("--gallery_annotation_path",
                        help="Path to the query annotations")

    args = parser.parse_args()

    return args


def cos_sim(x, y):
    return 1 - cosine(x, y)


def parse_annotation(img_data_root, annotation_path):
    image_paths = []
    ids = []
    with open(annotation_path) as f:
        for line in f.readlines():
            filename, id, _ = line.split(" ")  # cam_id is not needed
            image_paths.append(os.path.join(img_data_root, filename))
            ids.append(id)

    return image_paths, ids


def rank_gallery_features(query_feature, gallery_features, gallery_ids):
    cosines = [
        cos_sim(query_feature, gallery_feature)
        for gallery_feature in gallery_features
    ]
    gallery_info = list(zip(cosines, gallery_ids))
    gallery_info.sort(key=lambda x: x[0], reverse=True)
    sorted_ids = [x[1] for x in gallery_info]

    return sorted_ids


if __name__ == "__main__":
    args = parse_args()
    k = 5

    extractor = FeatureExtractor(model_name=args.model_name,
                                 model_path=args.model_path,
                                 device='cuda')

    query_filepaths, query_ids = parse_annotation(args.query_data_root,
                                                  args.query_annotation_path)
    gallery_filepaths, gallery_ids = parse_annotation(
        args.gallery_data_root, args.gallery_annotation_path)

    query_features = []
    gallery_features = []
    for query_image_path, gallery_image_path in tqdm(
            zip(query_filepaths, gallery_filepaths)):
        query_features.append(extractor(query_image_path)[0].cpu().tolist())
        gallery_features.append(
            extractor(gallery_image_path)[0].cpu().tolist())

    size = len(query_features)
    rank_res = [0] * k
    for feature, query_id in tqdm(zip(query_features, query_ids)):
        ranked_ids = rank_gallery_features(feature, gallery_features,
                                           gallery_ids)[:k]

        if query_id not in ranked_ids:  # query id is not in the top k of gallery data
            continue

        start = ranked_ids.index(query_id)
        for i in range(start, k):
            rank_res[i] += 1

    for i, res_value in enumerate(rank_res):
        print(f"Rank {i + 1}: {round(res_value / size, 3)}")

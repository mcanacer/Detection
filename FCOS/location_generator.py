import torch


class LocationGenerator(object):

    def __call__(self, feature_map_sizes, strides):
        locations = {}
        meshgrid = lambda x, y: [
            output.reshape(-1) for output in torch.meshgrid(x, y, indexing='ij')
        ]
        for idx, (level, stride) in enumerate(strides.items()):

            height, width = feature_map_sizes[idx]

            x, y = meshgrid(stride * (torch.arange(width) + 0.5), stride * (torch.arange(height) + 0.5))

            centers = torch.stack((x, y)).T  # [M, 2]

            locations[level] = centers

        return locations


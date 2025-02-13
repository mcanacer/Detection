import torch
import collections



class PointsGenerator(object):

    def __call__(self, feature_map_sizes, strides):
        locations = {}
        strides_list = []
        meshgrid = lambda x, y: [
            output.reshape(-1) for output in torch.meshgrid(x, y, indexing='xy')
        ]
        for idx, (level, stride) in enumerate(strides.items()):

            height, width = feature_map_sizes[idx]

            x, y = meshgrid(stride * (torch.arange(width) + 0.5), stride * (torch.arange(height) + 0.5))

            centers = torch.stack((x, y), dim=-1)  # [M, 2]

            strides_list.append(torch.full((height*width,), stride))

            locations[level] = centers

        strides = torch.cat(strides_list, dim=0)

        return locations, strides


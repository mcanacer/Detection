import torch
import collections



class PointsGenerator(object):

    def __call__(self, feature_map_sizes, strides):
        locations = {}
        strides_list = []
        meshgrid = lambda x, y: [
            output.reshape(-1) for output in torch.meshgrid(x, y, indexing='ij')
        ]
        for idx, (level, stride) in enumerate(strides.items()):

            height, width = feature_map_sizes[idx]

            x, y = meshgrid(stride * (torch.arange(width) + 0.5), stride * (torch.arange(height) + 0.5))

            centers = torch.stack((x, y)).T  # [M, 2]

            strides_list.append(torch.full((height*width,), stride))

            locations[level] = centers

        strides = torch.cat(strides_list, dim=0)

        return locations, strides


'''class PointsGenerator(object):

    def __call__(self, feature_map_sizes, feature_map_strides):
        points_list = []
        strides_list = []
        num_points = []

        meshgrid = lambda x, y: [
            output.reshape(-1) for output in torch.meshgrid(x, y, indexing='ij')
        ]

        for idx, (level, stride) in enumerate(feature_map_strides.items()):
            height, width = feature_map_sizes[idx]

            xs, ys = meshgrid((torch.arange(width, dtype=torch.float32) + 0.5), (torch.arange(height, dtype=torch.float32) + 0.5))

            points = torch.stack([xs, ys], dim=-1)  # [M, 2]
            strides = torch.full((height * width,), stride)  # [M]

            points_list.append(points)
            strides_list.append(strides)
            num_points.append(height * width)

        points = torch.cat(points_list, dim=0)
        strides = torch.cat(strides_list, dim=0)

        return points, strides, num_points
'''

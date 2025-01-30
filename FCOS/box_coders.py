import torch


class LTRB(object):

    def encode(self, boxes, points):
        '''
        Args:
            boxes: [N, M, 4]
            points [M, 2]
        Return
            targets: [N, M, 4]
        '''

        mins, maxs = torch.split(boxes, 2, dim=-1)

        indices = torch.unsqueeze(points, dim=0)

        lt = indices - mins
        rb = maxs - indices

        return torch.cat((lt, rb), dim=-1)


    def decode(self, preds, points):
        '''
        Args:
            preds: [N, M, 4]
            points: [M, 2]
        Return:
            targets: [N, M, 2]
        '''

        lt, rb = torch.split(preds, 2, dim=-1)

        indices = torch.unsqueeze(points, dim=0)

        mins = indices - lt
        maxs = rb + indices

        return torch.cat((mins, maxs), dim=-1)



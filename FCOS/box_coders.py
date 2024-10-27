import torch


class LTRB(object):

    def encode(self, boxes, locations):
        '''
        Args:
            boxes: [N, M, 4]
            locations [M, 2]
        Return
            targets: [N, M, 4]
        '''

        mins, maxs = torch.split(boxes, 2, dim=-1)

        lt = locations.unsqueeze(dim=0) - mins
        rb = maxs - locations.unsqueeze(dim=0)

        return torch.cat((lt, rb), dim=-1)


    def decode(self, preds, locations):
        '''
        Args:
            preds: [N, M, 4]
            locations: [M, 2]
        Return:
            targets: [N, M, 2]
        '''

        lt, rb = torch.split(preds, 2, dim=-1)

        mins = locations.unsqueeze(dim=0) - lt
        maxs = rb - locations.unsqueeze(dim=0)

        return torch.cat((mins, maxs), dim=-1)



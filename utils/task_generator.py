import random
from operator import itemgetter

import torch


class TaskGen:
    def __init__(self, task_size: int, inner_aug_count: int, outer_aug_count: int, batch_size: int, device: str):
        self.device = device
        self.task_size = task_size
        self.batch_size = batch_size
        self.inner_aug_count = inner_aug_count
        self.outer_aug_count = outer_aug_count
        self.inner_set = None
        self.outer_set = None
        self.rand_id = list(range(batch_size))

    def set_batch(self, batch: dict):
        x0, x1 = itemgetter("origs", "views")(batch)
        x = torch.cat([x0, x1], dim=1)
        self.inner_set, self.outer_set = x.split([self.inner_aug_count, self.outer_aug_count], dim=1)
        self.outer_set = torch.cat([x0, self.outer_set], dim=1)

    #
    # def sample(self):
    #     return self.inner_set
    def sample(self):
        batch_size = self.batch_size
        assert batch_size % self.task_size == 0
        selected_id = random.sample(self.rand_id, k=self.task_size, )
        unselected_id = list(set(self.rand_id) - set(selected_id))
        return self.inner_set[selected_id], self.inner_set[unselected_id]

    def queries(self):
        return self.outer_set

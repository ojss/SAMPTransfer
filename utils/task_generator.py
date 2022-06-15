import random
from operator import itemgetter

import torch


class TaskGen:
    def __init__(self, task_size: int, query_size: int, batch_size: int, device: str):
        self.device = device
        self.task_size = task_size
        self.batch_size = batch_size
        self.x = None
        rand_id = list(range(batch_size))
        self.rand_id_inner = rand_id[:batch_size - query_size]
        self.id_query = rand_id[: -query_size]

    def set_batch(self, batch: dict):
        x0, x1 = itemgetter("origs", "views")(batch)
        self.x = torch.cat([x0, x1], dim=1)

    def sample(self):
        batch_size = self.x.size(0)
        assert batch_size % self.task_size == 0
        selected_id = random.choices(self.rand_id_inner, k=self.task_size)
        return self.x[selected_id]

    def queries(self):
        return self.x[self.id_query]

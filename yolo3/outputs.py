import numpy as np
class Outputs:
    def __init__(self):
        self.num_outputs =0
        self.outputs = []
        self.masks = []
        self.num_masks = []

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current > self.num_outputs:
            raise StopIteration
        else:
            self.current += 1
            return self.get(self.current-1)

    def num(self):
        return self.num_outputs

    def size(self):
        if self.num_outputs > 0:
            return self.outputs[0].data.size(0)
        else:
            return 0

    def get(self, index):
        if index < self.num_outputs:
            return [self.outputs[index].data, self.masks[index], self.num_masks[index]]
        else:
            return [None, None, None]
    
    def get_out(self, index):
        if index < self.num_outputs:
            return self.outputs[index]
        else:
            return None

    def add(self, outbox):
        if len(outbox) == 3:
            self.outputs.append(outbox[0])
            self.masks.append(outbox[1])
            self.num_masks.append(outbox[2])
            self.num_outputs += 1
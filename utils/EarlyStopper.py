import copy
import torch
import torch.nn as nn
from collections import defaultdict

class EarlyStopper(object):

    def __init__(self, min_step, max_step, patience, improve = 'up'):
        self.improve = improve
        self.max_step = max_step
        self.best_scores = defaultdict(lambda:1000000000000 if improve=='down' else 0)
        self.best_models = defaultdict(lambda: None) # key: metric name; value: model
        self.increase_bool_tensor = defaultdict(lambda: torch.zeros(max_step).share_memory_())

        self.early_stop = torch.BoolTensor([False]).share_memory_()
        self.cur_step = 0
        self.min_step = max(min_step, patience)
        self.patience = patience
        self.best_epoch = 0

    def better(self, new_score, metric_name) -> bool:
        '''
        '''
        if self.improve == 'down':
            return new_score < self.best_scores[metric_name]
        
        if self.improve == 'up':
            return new_score > self.best_scores[metric_name]

    def update(self, step, updated_score:dict, updated_model):
        '''
        updated_score: dict key: metric name; value: score
        '''
        self.cur_step = step
        for metric_name, new_score in updated_score.items():
            if self.better(new_score=new_score, metric_name=metric_name):
                # print(f"{metric_name} gets better!")
                self.best_scores[metric_name] = new_score
                # deepcopy since updated_model is reference
                self.best_models[metric_name] = copy.deepcopy(updated_model)
                self.increase_bool_tensor[metric_name][step] = 1
                self.best_epoch = step
            else:
                # print(f"{metric_name} DOESN'T get better!")
                pass

        self.check_stop()

    def check_stop(self,):
        '''
        '''
        if self.cur_step <= self.min_step:
            self.early_stop[0] = False
        elif self.cur_step > self.min_step and self.cur_step <= self.max_step:
            # all metric not improved，-> early stop
            for metric_name, increase_bool_tensor in self.increase_bool_tensor.items():
                improved = increase_bool_tensor[(self.cur_step-self.patience+1): self.cur_step+1].sum() != 0
                if improved:
                    break
            else: # 
                self.early_stop[0] = True
        elif self.cur_step > self.max_step:
            self.early_stop[0] = True
        else:
            exit("wrong situation in early-stop check")

    @property
    def stopped(self):
        '''
        '''
        return self.early_stop.item()
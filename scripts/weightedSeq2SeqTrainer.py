import torch
from torch.nn import CrossEntropyLoss
import transformers
from transformers import Seq2SeqTrainer

class weightedSeq2SeqTrainer(Seq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs):
        
        labels = inputs.get("labels")
        outputs = model(**inputs)
        lm_logits = outputs.get('logits')

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        
        loss_fct_none = CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss_none = loss_fct_none(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        weighted_index = []
        for i in range(labels.view(-1).size(0)):
            if (labels.view(-1)[i] == 333 or labels.view(-1)[i] == 497) and labels.view(-1)[i+1] == 1:
                weighted_index.append(i)

        # different weighting schema
        global_step = self.state.global_step
        total_step = self.state.max_steps

        # weighting schema 1
        weight = 0.1 if (global_step / total_step) <= 0.6 else 0.0

        # weighting schema 2
        # if (global_step / total_step) <= 0.4:
        #     weight = (0.5 / (2*total_step)) * global_step
        # else:
        #     weight = (-0.5 / (3*total_step)) * global_step + (0.5 / 3)

        # weighting schema 3
        # if (global_step / total_step) <= 0.4:
        #     weight = 0.1
        # elif (global_step / total_step) <= 0.6:
        #     weight = (-0.5 / total_step) * global_step + 0.5
        # else:
        #     weight = 0.0
        
        for idx in weighted_index:
            loss = loss + weight*loss_none[idx]

        return loss

    def create_scheduler(self, num_training_steps, optimizer: torch.optim.Optimizer = None):
        
        self.lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer = self.optimizer, 
            num_warmup_steps = self.args.warmup_ratio * num_training_steps,
            num_training_steps = num_training_steps, 
            num_cycles = 3
        )
            
        return self.lr_scheduler
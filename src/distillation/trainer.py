import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, alpha=0.8, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        
        if self.teacher_model:
            self.teacher_model.eval()
            self.teacher_model.to(self.args.device)

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        # Forward pass student
        outputs_student = model(**inputs)
        logits_student = outputs_student.get("logits")
        
        # Forward pass teacher (with no grad)
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            logits_teacher = outputs_teacher.get("logits")
        
        # Calculate distillation loss (KL Divergence)
        # KLDivLoss expects input in log-probabilities and target in probabilities (or log-probs)
        # We use log_softmax for student and softmax for teacher
        
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        
        # Scale by temperature
        # Flatten to (batch * seq_len, vocab_size) so batchmean normalizes per-token
        flat_student = logits_student.view(-1, logits_student.size(-1))
        flat_teacher = logits_teacher.view(-1, logits_teacher.size(-1))
        
        loss_kd = (self.temperature ** 2) * loss_fct(
            F.log_softmax(flat_student / self.temperature, dim=-1),
            F.softmax(flat_teacher / self.temperature, dim=-1)
        )
        
        # Calculate standard task loss (Cross Entropy)
        # Hugging Face models usually return loss in outputs if labels are provided
        loss_ce = outputs_student.logits_loss if hasattr(outputs_student, "logits_loss") else outputs_student.loss
        
        if loss_ce is None:
             # Manually compute if not returned
             labels = inputs.get("labels")
             if labels is not None:
                 loss_ce = F.cross_entropy(logits_student.view(-1, logits_student.size(-1)), labels.view(-1))
             else:
                 loss_ce = 0.0

        # Combine losses
        loss = self.alpha * loss_kd + (1 - self.alpha) * loss_ce
        
        return (loss, outputs_student) if return_outputs else loss

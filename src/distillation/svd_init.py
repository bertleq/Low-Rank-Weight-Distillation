import torch
import torch.nn as nn
from modeling.low_rank_layer import LowRankLinear

def initialize_student_from_teacher(student_model, teacher_model):
    """
    Initializes the LowRankLinear layers of the student model using SVD on the teacher's weights.
    Assumes identical structure aside from the linear layers.
    """
    
    # Create mapping from name to module for teacher
    teacher_modules = dict(teacher_model.named_modules())
    
    for name, student_module in student_model.named_modules():
        if isinstance(student_module, LowRankLinear):
            if name in teacher_modules:
                teacher_module = teacher_modules[name]
                if isinstance(teacher_module, nn.Linear):
                    # Perform SVD initialization
                    with torch.no_grad():
                        # Get teacher weights
                        W = teacher_module.weight.data.float() # (out_features, in_features)
                        
                        # Compute SVD
                        try:
                            # Use low-rank SVD if possible for speed, but full SVD is safer for implementation
                            # We only need top 'rank' components
                            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                        except RuntimeError:
                            # Fallback for stability if needed (e.g. on CPU for large matrices)
                            print(f"Warning: SVD failed for layer {name}, using random initialization.")
                            continue
                        
                        rank = student_module.rank
                        
                        # Truncate to rank
                        U_r = U[:, :rank]
                        S_r = S[:rank]
                        Vh_r = Vh[:rank, :]
                        
                        # Compute square root of singular values for balanced distribution
                        sqrt_S_r = torch.diag(torch.sqrt(S_r))
                        
                        # Calculate new weights
                        # project_out (U part): (out_features, rank)
                        W_out = torch.mm(U_r, sqrt_S_r)
                        
                        # project_in (V part): (rank, in_features)
                        W_in = torch.mm(sqrt_S_r, Vh_r)
                        
                        # Assign weights
                        student_module.project_out.weight.data = W_out.to(student_module.project_out.weight.device).type(student_module.project_out.weight.dtype)
                        student_module.project_in.weight.data = W_in.to(student_module.project_in.weight.device).type(student_module.project_in.weight.dtype)
                        
                        # Copy bias if it exists
                        if teacher_module.bias is not None and student_module.project_out.bias is not None:
                            student_module.project_out.bias.data = teacher_module.bias.data.to(student_module.project_out.bias.device).type(student_module.project_out.bias.dtype)
                            
                    print(f"Initialized layer {name} with SVD (rank={rank})")
                else:
                    print(f"Warning: Matching module for {name} is not nn.Linear, skipping SVD init.")

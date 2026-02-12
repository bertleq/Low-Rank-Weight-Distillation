import sys
import os
import torch
import torch.nn as nn
import copy

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from modeling.low_rank_layer import LowRankLinear
from modeling.student import replace_linear_with_low_rank
from distillation.svd_init import initialize_student_from_teacher

def test_pipeline():
    print("Testing Pipeline...")
    
    # 1. Create dummy teacher (simple MLP)
    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(20, 5)
            
        def forward(self, x):
            return self.linear2(self.relu(self.linear1(x)))

    teacher = Teacher()
    
    # Initialize with recognizable weights for testing (e.g., identity-like or random)
    torch.manual_seed(42)
    teacher.linear1.weight.data = torch.randn(20, 10)
    teacher.linear1.bias.data = torch.randn(20)
    teacher.linear2.weight.data = torch.randn(5, 20)
    teacher.linear2.bias.data = torch.randn(5)
    
    print("Teacher created.")
    
    # 2. Create student
    student = copy.deepcopy(teacher) # Copy structure
    # Replace linear layers with rank 4
    # 10->20, rank 4. Params: 10*4 + 4*20 = 40 + 80 = 120. Teacher: 10*20 = 200.
    rank = 4
    replace_linear_with_low_rank(student, rank)
    
    print("Student created with LowRankLinear layers.")
    
    # Verify replacement
    assert isinstance(student.linear1, LowRankLinear)
    assert isinstance(student.linear2, LowRankLinear)
    print("Layer replacement verified.")
    
    # 3. Initialize with SVD
    initialize_student_from_teacher(student, teacher)
    print("SVD Initialization done.")
    
    # 4. Compare outputs
    x = torch.randn(2, 10)
    
    with torch.no_grad():
        y_teacher = teacher(x)
        y_student = student(x)
    
    diff = (y_teacher - y_student).abs().mean()
    print(f"Mean output difference after SVD init: {diff.item():.4f}")
    
    # 5. Compare with random init
    student_random = copy.deepcopy(student)
    def reset_params(m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()
    student_random.apply(reset_params)
    
    with torch.no_grad():
        y_random = student_random(x)
        
    diff_random = (y_teacher - y_random).abs().mean()
    print(f"Mean output difference with random init: {diff_random.item():.4f}")
    
    if diff < diff_random:
        print("SUCCESS: SVD initialization is closer to teacher than random initialization.")
    else:
        print("WARNING: SVD initialization did not improve over random (might happen for very small random matrices).")

if __name__ == "__main__":
    test_pipeline()

import torch

class LandmarkDistance:
    def __init__(self):
        pass
    
    def __call__(self, pred_landmarks, true_landmarks):
        return torch.mean(torch.norm(pred_landmarks - true_landmarks))

class LandmarkVelocityDifference:
    def __init__(self):
        pass
    
    def __call__(self, pred_landmarks, true_landmarks):
        pred_velocity = pred_landmarks[:, :, :, :1] - pred_landmarks[:, :, :, :-1]
        true_velocity = true_landmarks[:, :, :, :1] - true_landmarks[:, :, :, :-1]
        return torch.mean(torch.norm(pred_velocity - true_velocity))
    
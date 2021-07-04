import torch
import torch.functional as F

class TripletLoss(torch.nn.Module):
    """
    Triplet loss function.
    """

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):

        squarred_distance_1 = torch.linalg.norm(anchor - positive,dim=-1)
        
        squarred_distance_2 = torch.linalg.norm(anchor - negative,dim=-1)
        
        triplet_loss = torch.relu( self.margin + squarred_distance_1 - squarred_distance_2 ).mean()
        
        return triplet_loss
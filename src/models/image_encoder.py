import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=256, freeze_backbone=True, backbone_name='resnet18'):
        super().__init__()
        
        self.backbone_name = backbone_name
        
        if backbone_name == 'resnet18':
            try:
                from torchvision.models import resnet18, ResNet18_Weights
                self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            except ImportError:
                self.backbone = models.resnet18(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone_name == 'resnet34':
            try:
                from torchvision.models import resnet34, ResNet34_Weights
                self.backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
            except ImportError:
                self.backbone = models.resnet34(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Projection head to reduce dimension to output_dim
        self.projection = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        # If x is pre-extracted feature vectors (e.g. batch shape [batch, 512])
        if len(x.shape) == 2 and x.shape[1] == 512:
            return self.projection(x)
        features = self.backbone(x)
        return self.projection(features)

    def unfreeze_last_block(self):
        """
        Unfreezes the final layer/block of the ResNet for fine-tuning.
        """
        if self.backbone_name in ['resnet18', 'resnet34']:
            # Unfreeze layer4
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
        print(f"Unfroze last block of backbone {self.backbone_name}")

import torch
import torch.nn as nn
from src.models.image_encoder import ImageEncoder
from src.models.metadata_encoder import MetadataEncoder
from src.models.node_encoder import NODEEncoder
from src.models.heads import MultiTargetHead

class MultimodalModel(nn.Module):
    def __init__(self, 
                 meta_input_dim, 
                 image_emb_dim=256, 
                 meta_emb_dim=64, 
                 fusion_dim=128, 
                 num_targets=5,
                 use_image=True,
                 use_meta=True,
                 use_node=False,
                 freeze_backbone=True,
                 backbone_name='resnet18'):
        super().__init__()
        
        self.use_image = use_image
        self.use_meta = use_meta
        self.use_node = use_node
        
        # Encoders
        if self.use_image:
            self.image_encoder = ImageEncoder(
                output_dim=image_emb_dim, 
                freeze_backbone=freeze_backbone, 
                backbone_name=backbone_name
            )
        else:
            self.image_encoder = None
            
        if self.use_meta:
            if self.use_node:
                self.meta_encoder = NODEEncoder(
                    input_dim=meta_input_dim,
                    output_dim=meta_emb_dim
                )
            else:
                self.meta_encoder = MetadataEncoder(
                    input_dim=meta_input_dim, 
                    output_dim=meta_emb_dim
                )
        else:
            self.meta_encoder = None
            
        # Fusion projection layers
        if self.use_image and self.use_meta:
            concat_dim = image_emb_dim + meta_emb_dim
        elif self.use_image:
            concat_dim = image_emb_dim
        elif self.use_meta:
            concat_dim = meta_emb_dim
        else:
            raise ValueError("At least one of use_image or use_meta must be True")
            
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, fusion_dim),
            nn.ReLU()
        )
        
        # Prediction head
        self.head = MultiTargetHead(input_dim=fusion_dim, num_targets=num_targets)
        
    def forward(self, image, meta):
        # Forward pass for encoders
        embeddings = []
        
        if self.use_image:
            img_emb = self.image_encoder(image) # Shape: [batch, image_emb_dim]
            embeddings.append(img_emb)
            
        if self.use_meta:
            meta_emb = self.meta_encoder(meta) # Shape: [batch, meta_emb_dim]
            embeddings.append(meta_emb)
            
        # Fuse representations
        if len(embeddings) > 1:
            x = torch.cat(embeddings, dim=1) # Shape: [batch, concat_dim]
        else:
            x = embeddings[0]
            
        # Shared representation
        shared_rep = self.fusion(x) # Shape: [batch, fusion_dim]
        
        # Output predictions
        output = self.head(shared_rep) # Shape: [batch, num_targets]
        
        return output, shared_rep
        
    def unfreeze_backbone(self):
        """
        Unfreeze the last block of the image encoder backbone for fine-tuning.
        """
        if self.use_image and self.image_encoder is not None:
            self.image_encoder.unfreeze_last_block()

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel
from model_module.rope_attn import MutilHeadSelfAttn


class MaskedConv1d(nn.Conv1d):
    """Masked 1D convolution with automatic same-padding and optional input mask.
    Shapes:
      - Input: (B, L, C_in), input_mask: (B, L, 1) optional
      - Output: (B, L, C_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """Args:
            in_channels: input channels
            out_channels: output channels
            kernel_size: kernel width
            stride: stride
            dilation: dilation factor
            groups: groups for depth-wise conv
            bias: add learnable bias
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class AttnMean(nn.Module):
    def __init__(
        self,
        hidden_size,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.layer = MaskedConv1d(hidden_size, 1, 1)

    def forward(self, x, input_mask=None):
        x = self.layer_norm(x)
        batch_size = x.shape[0]
        attn = self.layer(x)
        attn = attn.view(batch_size, -1)  # [B, L]
        if input_mask is not None:
            attn = attn.masked_fill_(~input_mask.view(batch_size, -1).bool(), float("-inf"))
        attn = F.softmax(attn, dim=-1)  # [B, L]
        x = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        return x


class AttnTransform(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.layer = MaskedConv1d(hidden_size, 1, 1)

    def forward(self, x, input_mask=None):
        x = self.layer_norm(x)
        batch_size = x.shape[0]
        attn = self.layer(x)
        attn = attn.view(batch_size, -1)  # [B, L]
        if input_mask is not None:
            attn = attn.masked_fill_(~input_mask.view(batch_size, -1).bool(), float("-inf"))
        attn = F.softmax(attn, dim=-1).view(batch_size, -1, 1)  # [B, L, 1]
        x = attn * x  # [B, L, H]
        return x


class OutHead(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ac = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.re = nn.ReLU()
        self.linear = nn.Linear(hidden_dim // 2, out_dim)

    def forward(self, x):
        x = self.ac(self.fc1(x))
        x = self.dropout(x)
        x = self.re(self.fc2(x))
        x = self.linear(x)
        return x


class SeqBindModel(nn.Module):
    def __init__(self, args):
        super(SeqBindModel, self).__init__()
        self.config = args
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_heads
        self.dropout = args.dropout

        self.encoder = EsmModel.from_pretrained(args.model_locate)
        self.wt_ab_conv_transformer = AttnTransform(hidden_size=self.hidden_size)
        self.wt_ag_conv_transformer = AttnTransform(hidden_size=self.hidden_size)
        self.mt_ab_conv_transformer = AttnTransform(hidden_size=self.hidden_size)
        self.mt_ag_conv_transformer = AttnTransform(hidden_size=self.hidden_size)

        self.wt_ab_attn = MutilHeadSelfAttn(
            num_heads=self.num_heads, hidden_dim=self.hidden_size, dropout=self.dropout
        )
        self.mt_ab_attn = MutilHeadSelfAttn(
            num_heads=self.num_heads, hidden_dim=self.hidden_size, dropout=self.dropout
        )
        self.wt_ag_attn = MutilHeadSelfAttn(
            num_heads=self.num_heads, hidden_dim=self.hidden_size, dropout=self.dropout
        )
        self.mt_ag_attn = MutilHeadSelfAttn(
            num_heads=self.num_heads, hidden_dim=self.hidden_size, dropout=self.dropout
        )

        self.wt_ab_mean = AttnMean(self.hidden_size)
        self.mt_ab_mean = AttnMean(self.hidden_size)
        self.wt_ag_mean = AttnMean(self.hidden_size)
        self.mt_ag_mean = AttnMean(self.hidden_size)

        self.bn = nn.BatchNorm1d(self.hidden_size * 2)

        self.out_head = OutHead(self.hidden_size * 2, 1)

        self.w_ab_head = nn.Linear(self.hidden_size, 33) # ESM2 vocab size
        self.w_ag_head = nn.Linear(self.hidden_size, 33)
        self.m_ab_head = nn.Linear(self.hidden_size, 33)
        self.m_ag_head = nn.Linear(self.hidden_size, 33)

        if self.config.freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
    @torch.no_grad()
    def get_encoder_embeddings(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def forward(
        self,
        wt_ab_inputs_ids,  
        wt_ab_inputs_mask, 
        mut_ab_inputs_ids, 
        mt_ab_inputs_mask, 
        wt_ag_inputs_ids, 
        wt_ag_inputs_mask, 
        mut_ag_inputs_ids, 
        mt_ag_inputs_mask, 
        wt_features=None,
        mt_features=None,
        labels=None,
        return_attn=False,
        with_value=False,
        output_embeddings=False,
    ):
        # Encoder
        wt_ab_embeddings = self.get_encoder_embeddings(
            input_ids=wt_ab_inputs_ids, attention_mask=wt_ab_inputs_mask
        )  # [B, L, H]
        mt_ab_embeddings = self.get_encoder_embeddings(
            input_ids=mut_ab_inputs_ids, attention_mask=mt_ab_inputs_mask
        )  # [B, L, H]

        wt_ag_embeddings = self.get_encoder_embeddings(
            input_ids=wt_ag_inputs_ids, attention_mask=wt_ag_inputs_mask
        )
        mt_ag_embeddings = self.get_encoder_embeddings(
            input_ids=mut_ag_inputs_ids, attention_mask=mt_ag_inputs_mask
        )
        
        wt_ab_embeddings = self.wt_ab_conv_transformer(
            wt_ab_embeddings, wt_ab_inputs_mask
        )
        mt_ab_embeddings = self.mt_ab_conv_transformer(
            mt_ab_embeddings, mt_ab_inputs_mask
        )

        wt_ag_embeddings = self.wt_ag_conv_transformer(
            wt_ag_embeddings, wt_ag_inputs_mask
        )
        mt_ag_embeddings = self.mt_ag_conv_transformer(
            mt_ag_embeddings, mt_ag_inputs_mask
        )

        wt_ag_embedding = self.wt_ag_attn(
            wt_ag_embeddings, wt_ab_embeddings, wt_ab_embeddings, mask=wt_ab_inputs_mask
        )  # [B, L_q, H]

        mt_ag_embedding = self.mt_ag_attn(
            mt_ag_embeddings, mt_ab_embeddings, mt_ab_embeddings, mask=mt_ab_inputs_mask
        )  # [B, L_q, H]

        wt_ab_embedding = self.wt_ab_attn(
            wt_ab_embeddings, wt_ag_embeddings, wt_ag_embeddings, mask=wt_ag_inputs_mask
        )
        mt_ab_embedding = self.mt_ab_attn(
            mt_ab_embeddings, mt_ag_embeddings, mt_ag_embeddings, mask=mt_ag_inputs_mask
        )
        mt_ab_embedding_out = mt_ab_embedding.clone()

        wt_ab_logits = self.w_ab_head(wt_ab_embedding)  # [B, L, 33]
        wt_ag_logits = self.w_ag_head(wt_ag_embedding)  # [B, L, 33]
        mt_ab_logits = self.m_ab_head(mt_ab_embedding)  # [B, L, 33]
        mt_ag_logits = self.m_ag_head(mt_ag_embedding)  # [B, L, 33]

        wt_ag_embedding = self.wt_ag_mean(wt_ag_embedding, wt_ag_inputs_mask)  # [B, H]
        mt_ag_embedding = self.mt_ag_mean(mt_ag_embedding, mt_ag_inputs_mask)  # [B, H]
        wt_ab_embedding = self.wt_ab_mean(wt_ab_embedding, wt_ab_inputs_mask)
        mt_ab_embedding = self.mt_ab_mean(mt_ab_embedding, mt_ab_inputs_mask)

        wt_abag = wt_ag_embedding + wt_ab_embedding  # [B, H]
        mt_abag = mt_ag_embedding + mt_ab_embedding  # [B, H]

        rep = torch.cat([wt_abag, mt_abag], dim=1)  # [B, 2H]
        rep = self.bn(rep)
        out = self.out_head(rep)

        if with_value:
            value = self.value_head(rep)  # [B, 1]
            if output_embeddings == True:
                return out.squeeze(1), wt_ab_logits, wt_ag_logits, mt_ab_logits, mt_ag_logits, mt_ab_embedding_out, value
            else:
                return out.squeeze(1), wt_ab_logits, wt_ag_logits, mt_ab_logits, mt_ag_logits, value

        elif with_value == False and output_embeddings == True:
            return out.squeeze(1), wt_ab_logits, wt_ag_logits, mt_ab_logits, mt_ag_logits, mt_ab_embedding_out
        else:
            return out.squeeze(1), wt_ab_logits, wt_ag_logits, mt_ab_logits, mt_ag_logits  # [B], [B, L, 33] x4

def freeze_for_mutation_finetune(model: SeqBindModel, num_unfreeze_layers: int = 4):
    # 1) Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # 2) Unfreeze the last N layers of ESM backbone
    total_layers = len(model.encoder.encoder.layer)  # e.g., 33
    for idx, block in enumerate(model.encoder.encoder.layer):
        if idx >= total_layers - num_unfreeze_layers:
            for p in block.parameters():
                p.requires_grad = True

    # 3) Keep all wt_* branches frozen
    for name, module in model.named_modules():
        if name.startswith("wt_"):
            for p in module.parameters():
                p.requires_grad = False

    # 4) Unfreeze all mt_* branches
    for name, module in model.named_modules():
        if name.startswith("mt_"):
            for p in module.parameters():
                p.requires_grad = True

    # 5) Keep BN and OutHead trainable
    for p in model.bn.parameters():
        p.requires_grad = True
    for p in model.out_head.parameters():
        p.requires_grad = True

def freeze_for_mutation_finetune_v2(model: SeqBindModel, config: dict = None):
    """Improved freezing strategies with multiple presets.
    Args:
        model: SeqBindModel instance
        config: dict with optional keys:
            - strategy: 'balanced' | 'mutation_focused' | 'attention_only' | 'progressive'
            - num_unfreeze_layers: number of ESM layers to unfreeze
            - unfreeze_embeddings: whether to unfreeze ESM embeddings
    """
    if config is None:
        config = {
            'strategy': 'balanced',
            'num_unfreeze_layers': 4,
            'unfreeze_embeddings': False
        }
    
    strategy = config.get('strategy', 'balanced')
    num_unfreeze_layers = config.get('num_unfreeze_layers', 4)
    unfreeze_embeddings = config.get('unfreeze_embeddings', False)
    
    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False
    
    if strategy == 'balanced':
        # Strategy 1: balanced training (recommended)
        # Unfreeze last few ESM layers
        total_layers = len(model.encoder.encoder.layer)
        for idx, block in enumerate(model.encoder.encoder.layer):
            if idx >= total_layers - num_unfreeze_layers:
                for p in block.parameters():
                    p.requires_grad = True
        
        # Optional: unfreeze embeddings to learn special tokens
        if unfreeze_embeddings:
            for p in model.encoder.embeddings.parameters():
                p.requires_grad = True
        
        # Unfreeze attention/transform and pooling modules
        attention_modules = [
            'wt_ab_conv_transformer', 'wt_ag_conv_transformer',
            'mt_ab_conv_transformer', 'mt_ag_conv_transformer',
            'wt_ab_attn', 'wt_ag_attn', 'mt_ab_attn', 'mt_ag_attn',
            'wt_ab_mean', 'wt_ag_mean', 'mt_ab_mean', 'mt_ag_mean'
        ]
        
        for name, module in model.named_modules():
            if any(name == module_name for module_name in attention_modules):
                for p in module.parameters():
                    p.requires_grad = True
        
        # Unfreeze heads
        for p in model.bn.parameters():
            p.requires_grad = True
        for p in model.out_head.parameters():
            p.requires_grad = True
        
        # Unfreeze mutation-related classifier heads only
        for p in model.m_ab_head.parameters():
            p.requires_grad = True
        for p in model.m_ag_head.parameters():
            p.requires_grad = True
            
    elif strategy == 'mutation_focused':
        # Strategy 2: mutation focused
        total_layers = len(model.encoder.encoder.layer)
        for idx, block in enumerate(model.encoder.encoder.layer):
            if idx >= total_layers - num_unfreeze_layers:
                for p in block.parameters():
                    p.requires_grad = True
        
        # Unfreeze mutation branches
        mt_modules = ['mt_ab_conv_transformer', 'mt_ag_conv_transformer',
                      'mt_ab_attn', 'mt_ag_attn', 'mt_ab_mean', 'mt_ag_mean']
        
        for name, module in model.named_modules():
            if any(name == module_name for module_name in mt_modules):
                for p in module.parameters():
                    p.requires_grad = True
        
        # Also update wt_* cross-attention modules
        for p in model.wt_ab_attn.parameters():
            p.requires_grad = True
        for p in model.wt_ag_attn.parameters():
            p.requires_grad = True
        
        # Unfreeze heads
        for p in model.bn.parameters():
            p.requires_grad = True
        for p in model.out_head.parameters():
            p.requires_grad = True
        for p in model.m_ab_head.parameters():
            p.requires_grad = True
        for p in model.m_ag_head.parameters():
            p.requires_grad = True

    elif strategy == 'light':
        total_layers = len(model.encoder.encoder.layer)
        for idx, block in enumerate(model.encoder.encoder.layer):
            if idx >= total_layers - num_unfreeze_layers:
                for p in block.parameters():
                    p.requires_grad = True

        # Unfreeze heads
        for p in model.bn.parameters():
            p.requires_grad = True
        for p in model.out_head.parameters():
            p.requires_grad = True
        for p in model.m_ab_head.parameters():
            p.requires_grad = True
        for p in model.m_ag_head.parameters():
            p.requires_grad = True
            
    elif strategy == 'attention_only':
        # Strategy 3: attention-only lightweight finetuning (keep ESM frozen)
        attention_modules = [
            'wt_ab_attn', 'wt_ag_attn', 'mt_ab_attn', 'mt_ag_attn',
            'wt_ab_mean', 'wt_ag_mean', 'mt_ab_mean', 'mt_ag_mean'
        ]
        
        for name, module in model.named_modules():
            if any(name == module_name for module_name in attention_modules):
                for p in module.parameters():
                    p.requires_grad = True
        
        # Unfreeze heads
        for p in model.bn.parameters():
            p.requires_grad = True
        for p in model.out_head.parameters():
            p.requires_grad = True
        for p in model.m_ab_head.parameters():
            p.requires_grad = True
        for p in model.m_ag_head.parameters():
            p.requires_grad = True
            
    elif strategy == 'progressive':
        # Strategy 4: progressive unfreezing (initial setup)
        total_layers = len(model.encoder.encoder.layer)
        initial_unfreeze = min(2, num_unfreeze_layers)
        for idx, block in enumerate(model.encoder.encoder.layer):
            if idx >= total_layers - initial_unfreeze:
                for p in block.parameters():
                    p.requires_grad = True
        
        # Unfreeze downstream modules
        downstream_modules = [
            'wt_ab_conv_transformer', 'wt_ag_conv_transformer',
            'mt_ab_conv_transformer', 'mt_ag_conv_transformer',
            'wt_ab_attn', 'wt_ag_attn', 'mt_ab_attn', 'mt_ag_attn',
            'wt_ab_mean', 'wt_ag_mean', 'mt_ab_mean', 'mt_ag_mean',
            'bn', 'out_head', 'm_ab_head', 'm_ag_head'
        ]
        
        for name, module in model.named_modules():
            if any(name.startswith(module_name) for module_name in downstream_modules):
                for p in module.parameters():
                    p.requires_grad = True
    
    # Ensure value_head remains trainable if present
    if hasattr(model, 'value_head'):
        for p in model.value_head.parameters():
            p.requires_grad = True
    
    # Print trainable parameter stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")


def progressive_unfreeze(model: SeqBindModel, epoch: int, total_epochs: int, max_unfreeze_layers: int = 6):
    """Helper for progressive unfreezing during training.
    Args:
        model: model instance
        epoch: current epoch (0-based)
        total_epochs: total epochs
        max_unfreeze_layers: maximum layers to unfreeze
    """
    # Compute current number of layers to unfreeze
    progress = epoch / total_epochs
    current_unfreeze = int(2 + progress * (max_unfreeze_layers - 2))
    
    # Refreeze all ESM layers
    for p in model.encoder.parameters():
        p.requires_grad = False
    
    # Unfreeze the last N layers
    total_layers = len(model.encoder.encoder.layer)
    for idx, block in enumerate(model.encoder.encoder.layer):
        if idx >= total_layers - current_unfreeze:
            for p in block.parameters():
                p.requires_grad = True
    
    print(f"Epoch {epoch}: Unfrozen last {current_unfreeze} layers of ESM")
import torch
import torch.nn.functional as F
from torch import nn


class SchedulerTransformerBlock(nn.Module):
    def __init__(
        self,
        i,
        image_encoder_depth,
        image_encoder_width,
        text_embed_dim,
        cross_attention_heads,
        attention_dim,
    ):
        super().__init__()

        self.image_encoder_depth = image_encoder_depth
        self.image_encoder_width = image_encoder_width

        # Каналы для этого transformer блока (зависят только от i, не от j)
        channels = image_encoder_width * 2 ** min(4, i)
        
        # Создаем conv_stack где все слои работают с одинаковым количеством каналов
        conv_blocks = []
        for j in range(image_encoder_depth):
            conv_blocks.append(nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1  # добавляем padding чтобы сохранить размер
                ),
                nn.GroupNorm(32, channels),
                nn.SiLU()
            ))
        
        self.conv_stack = nn.Sequential(*conv_blocks)

        self.query_projection = nn.Linear(channels, channels)
        self.key_value_projection = nn.Linear(text_embed_dim, channels)

        self.mha = nn.MultiheadAttention(channels, cross_attention_heads, batch_first=True)

    def forward(self, x, e_text):
        x = self.conv_stack(x)
        B, C, H, W = x.shape
        # Правильная перестановка для batch_first=True
        x_flat = x.reshape(B, C, H * W).permute(0, 2, 1)  # B x (H*W) x C
        x_query = self.query_projection(x_flat)
        x_text = self.key_value_projection(e_text)

        attn_out, _ = self.mha(x_query, x_text, x_text)
        x_flat = x_flat + attn_out
        x_flat = F.layer_norm(x_flat, [C])
        x = x_flat.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class ImageEncoder(nn.Module):
    def __init__(
        self,
        image_encoder_depth,
        image_encoder_width,
        text_embed_dim,
        cross_attention_heads,
        attention_dim,
        number_of_transformer_blocks,
        input_channels=None
    ):
        super().__init__()
           
        if input_channels is None:
            input_channels = image_encoder_width
        
        self.input_proj = None
        if input_channels != image_encoder_width:
            self.input_proj = nn.Conv2d(
                input_channels, 
                image_encoder_width, 
                kernel_size=1
            ) 
        
        self.transformer_blocks = nn.ModuleList([])
        for i in range(number_of_transformer_blocks):
            # Добавляем transformer block
            self.transformer_blocks.append(
                SchedulerTransformerBlock(
                    i, 
                    image_encoder_depth, 
                    image_encoder_width, 
                    text_embed_dim, 
                    cross_attention_heads, 
                    attention_dim
                )
            )
            
            # Добавляем downsampling слой между блоками (кроме последнего)
            if i < number_of_transformer_blocks - 1:
                in_channels = image_encoder_width * 2 ** min(4, i)
                out_channels = image_encoder_width * 2 ** min(4, i + 1)
                self.transformer_blocks.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, out_channels),
                    nn.SiLU()
                ))

    def forward(self, x, e_text):
        if self.input_proj is not None:
            x = self.input_proj(x)
            
        xs = []
        for i, layer in enumerate(self.transformer_blocks):
            if i % 2 == 0:
                # Transformer block
                x = layer(x, e_text)
                xs.append(F.avg_pool2d(x, x.shape[-2:])[..., 0, 0])
            else:
                # Downsampling block
                x = layer(x)
        xs = torch.cat(xs, dim=1)
        return xs
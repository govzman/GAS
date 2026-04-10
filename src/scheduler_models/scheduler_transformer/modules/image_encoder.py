import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        depth,
        image_encoder_width
    ): 
        in_channels = image_encoder_width * 2 ** min(4, depth)
        out_channels = image_encoder_width * 2 ** min(4, depth + 1)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3
            ),
            nn.GroupNorm(32, out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x)

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

        self.conv_stack = nn.ModuleList([ConvBlock(i + j, image_encoder_width) for j in range(image_encoder_depth)])

        out_channels = self.conv_stack[-1][0].out_channels
        self.query_projection = nn.Linear(out_channels, out_channels)
        self.key_value_projection = nn.Linear(text_embed_dim, out_channels)

        self.mha = nn.MultiheadAttention(out_channels, cross_attention_heads, batch_first=True)

    def forward(self, x, e_text):
        x = self.conv_stack(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        x_query = self.query_projection(x)
        x_text = self.key_value_projection(e_text)

        x = x + self.mha(x_query, x_text, x_text)
        x = F.layer_norm(x, [C, H * W]).reshape(B, C, H, W)
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
    ):
        super().__init__()

        self.transformer_blocks = nn.ModuleList([])
        for i in range(number_of_transformer_blocks):
            self.transformer_blocks.append(SchedulerTransformerBlock(i, image_encoder_depth, image_encoder_width, text_embed_dim, cross_attention_heads))
            in_channels = image_encoder_width * 2 ** min(4, i * number_of_transformer_blocks - 1)
            out_channels = image_encoder_width * 2 ** min(4, (i + 1) * number_of_transformer_blocks - 1)
            if i < number_of_transformer_blocks - 1:
                self.transformer_blocks.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
                    nn.GroupNorm(32, out_channels),
                    nn.SiLU()
                ))

    def forward(self, x, e_text):
        xs = []
        for i, layer in enumerate(self.transformer_blocks):
            if i % 2 == 0:
                x = layer(x, e_text)
                xs.append(F.avg_pool2d(x, x.shape[-2:])[..., 0, 0])
            else:
                x = layer(x)
        xs = torch.cat(xs, dim=1)
        return xs

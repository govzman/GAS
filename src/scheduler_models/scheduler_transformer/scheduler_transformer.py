import torch
import torch.nn.functional as F
from torch import nn

from src.scheduler_models.scheduler_transformer.modules import MLP, ImageEncoder


class SchedulerTransformer(nn.Module):
    """
    https://arxiv.org/pdf/2511.22177
    """

    def __init__(
        self,
        image_encoder_depth,
        image_encoder_width,
        text_embed_dim,
        cross_attention_heads,
        attention_dim,
        number_of_transformer_blocks,
        num_mlp_layers,
        hidden_dim,
        num_timesteps
    ):
        """
        Args:
            encoder_channels (int): number of channels in audio encoder.
        """
        super().__init__()

        final_xs_dim = sum(self.image_encoder_width * 2 ** min(4, i + self.image_encoder_depth) for i in range(self.number_of_transformer_blocks))
        self.image_encoder = ImageEncoder(
            image_encoder_depth,
            image_encoder_width,
            text_embed_dim,
            cross_attention_heads,
            attention_dim,
            number_of_transformer_blocks
        )
        self.mlps = MLP(
            num_mlp_layers,
            final_xs_dim,
            hidden_dim,
            num_timesteps
        )

    def forward(self, x, e_text, **batch):
        """
        x (tensor): B x H x W x C - image shaped noise
        e_text (tensor): B x L_text x d_text - token text emb
        """

        B, H, W, C = x.shape
        return self.mlps(self.image_encoder(x.reshape(B, C, H, W), e_text))

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
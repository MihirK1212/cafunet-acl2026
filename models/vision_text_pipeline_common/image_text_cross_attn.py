import torch
import torch.nn as nn

class ImageTextCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn_image_by_text = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_text_by_image = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, image_feats, text_feats):
        # image_feats, text_feats: (batch_size, embed_dim)

        # Reshape to (batch_size, seq_len=1, embed_dim) for attention
        image_feats = image_feats.unsqueeze(1)
        text_feats = text_feats.unsqueeze(1)

        # Image attended by text (Query = image, Key/Value = text)
        img_attn_output, _ = self.attn_image_by_text(query=image_feats, key=text_feats, value=text_feats)

        # Text attended by image (Query = text, Key/Value = image)
        txt_attn_output, _ = self.attn_text_by_image(query=text_feats, key=image_feats, value=image_feats)

        # Squeeze seq_len dim
        img_attn_output = img_attn_output.squeeze(1)
        txt_attn_output = txt_attn_output.squeeze(1)

        return img_attn_output, txt_attn_output

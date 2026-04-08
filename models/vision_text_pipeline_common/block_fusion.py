import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockFusion(nn.Module):
    def __init__(self, dim_x, dim_y, output_dim, num_blocks=4, rank=128):
        super(BlockFusion, self).__init__()
        self.num_blocks = num_blocks
        self.rank = rank
        self.output_dim = output_dim

        # Project x and y into multiple blocks
        self.proj_x = nn.ModuleList([nn.Linear(dim_x, rank, bias=False) for _ in range(num_blocks)])
        self.proj_y = nn.ModuleList([nn.Linear(dim_y, rank, bias=False) for _ in range(num_blocks)])

        # Final linear layer to reduce concatenated output
        self.output_layer = nn.Linear(num_blocks * rank, output_dim)

    def forward(self, x, y):
        # x: (batch_size, dim_x)
        # y: (batch_size, dim_y)

        block_outputs = []

        for i in range(self.num_blocks):
            x_proj = F.relu(self.proj_x[i](x))  # Apply ReLU after projection
            y_proj = F.relu(self.proj_y[i](y))

            block = x_proj * y_proj     # Element-wise product (Hadamard)
            block_outputs.append(block)

        # Concatenate blocks
        concat = torch.cat(block_outputs, dim=1)  # (batch_size, num_blocks * rank)

        out = self.output_layer(concat)  # (batch_size, output_dim)
        out = F.relu(out)  # Optional: activate final output

        return out

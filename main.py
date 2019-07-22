class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv_block_1 = nn.sequential(
            nn.Conv2d(1,4,kernel_size=5, stride=2),
            nn.batchnorm(4),
            nn.Relu()
        )    
        self.conv_block_2 = nn.sequential(
            nn.Conv2d(4,8,kernel_size=5, stride=2),
            nn.batchnorm(8),
            nn.Relu()
        )    
        self.conv_block_3 = nn.sequential(
            nn.Conv2d(8,16,kernel_size=5, stride=2),
            nn.batchnorm(16),
            nn.Relu()
        )    
        self.conv_block_4 = nn.sequential(
            nn.Conv2d(16,32,kernel_size=5, stride=2),
            nn.batchnorm(4),
            nn.Relu()
        )    
        self.conv_block_5 = nn.sequential(
            nn.Conv2d(32,64,kernel_size=5, stride=2),
            nn.Tanh()
        )    

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
from torch import nn
import torch


cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1,4,kernel_size=5, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )    
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(4,8,kernel_size=5, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )    
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(8,16,kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )    
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=5, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )    
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=5, stride=2),
            nn.Tanh()
        )    

    def forward(self, nosie_z):
        out = self.conv_block_1(z)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        out = self.conv_block_4(out)
        out = self.conv_block_5(out)
        out = out.view(-1, 64*64*3)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.dconv_1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.dconv_2 = nn.Sequential(
            nn.Conv2d(64, 32,kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.dconv_3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        self.dconv_4 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=5, stride=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU()
        )

        self.adv_layer = nn.Sequential(nn.Linear(8 * 8 ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.dconv_1(img)
        out = self.dconv_2(out)
        out = self.dconv_3(out)
        out = self.dconv_4(out)

        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
        

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
from torch import nn
import torch
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import torchvision.utils as vutils
import torch.nn.parallel
import torch.optim as optim

BATCH_SIZE = 128
DATASET_ROOT_DIR = r'C:\Users\CORE\Desktop\miqdude\Kawung'
# 'C:\Users\CORE\Desktop\miqdude\Kawung'
LEARNING_RATE = 0.002
EPOCH = 2000
SAVE_INTERVAL = 400
NOISE_VECTOR_DIM = 100


cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(100,1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(True)
        )    
        self.conv_block_2 = nn.Sequential(
            nn.ConvTranspose2d(1024,512,kernel_size=5),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )    
        self.conv_block_3 = nn.Sequential(
            nn.ConvTranspose2d(512,256,kernel_size=5, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )    
        self.conv_block_4 = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )    
        self.conv_block_5 = nn.Sequential(
            nn.ConvTranspose2d(128,3,kernel_size=5, stride=2),            
            nn.Tanh()
        )

    def forward(self, noise_z):
        out = self.conv_block_1(noise_z)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        out = self.conv_block_4(out)
        out = self.conv_block_5(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(128, 256,kernel_size=5, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=5, stride=2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace = True),
        )

    def forward(self, img):
        out = self.dconv_1(img)
        out = self.dconv_2(out)
        out = self.dconv_3(out)
        out = self.dconv_4(out)

        out = out.view(-1)

        return out
        

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
dataloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        DATASET_ROOT_DIR,
        transform = transforms.Compose(
            [transforms.Resize((128,128)), transforms.ToTensor()]
        )
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


for epoch in range(EPOCH):

    for i, data in enumerate(dataloader, 0):
        
        netD.zero_grad()
        
        real, _ = data
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0]))
        output = netD(input)
        errD_real = criterion(output, target)
        
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = netD(fake.detach())
        errD_fake = criterion(output, target)
        
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = netD(fake)
        errG = criterion(output, target)
        errG.backward()
        optimizerG.step()
        

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0]))
        if i % 100 == 0:
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True)
            fake = netG(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)
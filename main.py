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

        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)
        

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
            [transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Optimizers
optimizerG = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


for epoch in range(EPOCH):

    for i, data in enumerate(dataloader, 0):

        # Training discriminator with real images
        
        discriminator.zero_grad() # clears the gradient
        
        real, _ = data
        input = Variable(real.type(Tensor))  # input images must be Tensor for gpu compute
        target = Variable(torch.ones(input.size()[0]).cuda())
        output = discriminator(input)
        errD_real = adversarial_loss(output, target)
        

        # Generator generate images
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1).cuda())
        fake = generator(noise)
        target = Variable(torch.zeros(input.size()[0]).cuda())
        
        # discriminator estimates fake images
        output = discriminator(fake.detach())
        errD_fake = adversarial_loss(output, target)
        
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        generator.zero_grad()
        target = Variable(torch.ones(input.size()[0]).cuda())
        output = discriminator(fake)
        errG = adversarial_loss(output, target)
        errG.backward()
        optimizerG.step()
        

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data, errG.data))
        if i % 100 == 0:
            vutils.save_image(real, '%s/real_samples_%03d.png' % ("./results",epoch), normalize = True)
            fake = generator(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)
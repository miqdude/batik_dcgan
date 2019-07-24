from torch import nn
import torch
from torchvision import datasets
from torchvision import transforms



BATCH_SIZE = 128
DATASET_ROOT_DIR = 'root_folder_dir'
LEARNING_RATE = 0.002
EPOCH = 2000

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

        self.adv_layer = nn.Sequential(nn.Linear(8 * 8 * 8, 1), nn.Sigmoid())

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

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
dataloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        DATASET_ROOT_DIR,
        transforms.Resize(
            (128,128)
        )
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------

for epoch in range(EPOCH):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)


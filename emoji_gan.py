from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer, BertModel
import numpy as np
from torchvision.utils import save_image
from PIL import Image
import json
import os

# Text Encoder using BERT
class TextEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, output_dim)  # BERT hidden size is 768
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        text_features = self.fc(pooled_output)
        return text_features

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, text_embedding_dim=256, channels=4):
        super(Generator, self).__init__()
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, 256),
            nn.ReLU()
        )
        
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU()
        )
        
        # Initial size: 4x4
        self.init_size = 4
        self.fc = nn.Linear(512, self.init_size ** 2 * 512)
        
        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z, text_embedding):
        text_features = self.text_projection(text_embedding)
        latent_features = self.latent_projection(z)
        
        # Concatenate text and latent features
        combined_features = torch.cat([text_features, latent_features], dim=1)
        
        # Project and reshape
        x = self.fc(combined_features)
        x = x.view(-1, 512, self.init_size, self.init_size)
        
        # Generate image
        img = self.conv_blocks(x)
        return img

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, text_embedding_dim=256, channels=4):
        super(Discriminator, self).__init__()
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, 256),
            nn.LeakyReLU(0.2)
        )
        
        self.conv_blocks = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 8x8 -> 4x4
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4 + 256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img, text_embedding):
        img_features = self.conv_blocks(img)
        img_features = img_features.view(img_features.size(0), -1)
        
        text_features = self.text_projection(text_embedding)
        
        # Concatenate image and text features
        combined_features = torch.cat([img_features, text_features], dim=1)
        
        # Compute validity
        validity = self.fc(combined_features)
        return validity

# Custom Dataset
class EmojiDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None, max_length=32):
        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length
        
        # Load captions
        with open(captions_file, 'r') as f:
            self.captions = json.load(f)
            
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def __len__(self):
        return len(self.captions)
        
    def __getitem__(self, idx):
        item = self.captions[idx]
        image_path = os.path.join(self.image_dir, str(item['index'])) + ".png"
        caption = item['caption']
        
        # Load and transform image
        image = Image.open(image_path).convert('RGBA')
        if self.transform:
            image = self.transform(image)
            
        # Tokenize text
        encoding = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'caption': caption
        }

# Training functions
class EmojiGAN:
    def __init__(self, device='cuda'):
        self.device = device
        self.latent_dim = 100
        self.text_embedding_dim = 256
        
        # Initialize models
        self.text_encoder = TextEncoder(self.text_embedding_dim).to(device)
        self.generator = Generator(self.latent_dim, self.text_embedding_dim).to(device)
        self.discriminator = Discriminator(self.text_embedding_dim).to(device)
        
        # Freeze BERT parameters
        for param in self.text_encoder.bert.parameters():
            param.requires_grad = False
            
        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.criterion = nn.BCELoss()
        
    def train_step(self, batch):
        real_images = batch['image'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        batch_size = real_images.size(0)
        
        # Ground truths
        valid = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)
        
        # Get text embeddings
        with torch.no_grad():
            text_embeddings = self.text_encoder(input_ids, attention_mask)
            
        # -----------------
        #  Train Generator
        # -----------------
        
        self.g_optimizer.zero_grad()
        
        # Generate images
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        gen_imgs = self.generator(z, text_embeddings)
        
        # Loss measures generator's ability to fool the discriminator
        g_loss = self.criterion(self.discriminator(gen_imgs, text_embeddings), valid)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        self.d_optimizer.zero_grad()
        
        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.criterion(self.discriminator(real_images, text_embeddings), valid)
        fake_loss = self.criterion(self.discriminator(gen_imgs.detach(), text_embeddings), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        self.d_optimizer.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'gen_imgs': gen_imgs
        }

    def save_tensor_as_image(self, tensor, filepath):
        """Save a tensor as an image with alpha channel"""
        # Unnormalize the tensor
        # tensor = (tensor + 1) / 2.0
        
        # # Convert to PIL image
        # if tensor.shape[0] == 4:  # RGBA
        #     # Separate RGB and Alpha channels
        #     rgb = tensor[:3]
        #     alpha = tensor[3:]
            
        #     # Save RGB with alpha
        #     save_image(tensor, filepath, normalize=False)
        # else:  # RGB
        #     save_image(tensor, filepath, normalize=False)  
        save_image(tensor, filepath, normalize=False) 

        print(f"Saved {filepath}")  
    
    def generate_emoji(self, caption, epoch, num_samples=1, save_dir="emoji-gan-modified/output"):
        self.generator.eval()
        
        # Tokenize caption
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoding = tokenizer(
            caption,
            max_length=32,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get text embeddings
        with torch.no_grad():
            text_embeddings = self.text_encoder(input_ids, attention_mask)
            text_embeddings = text_embeddings.repeat(num_samples, 1)
            
            # Generate images
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            gen_imgs = self.generator(z, text_embeddings)

        # Save images
        # saved_paths = []
        
        
        filename = f'emoji_{epoch+1}.png'
        filepath = os.path.join(save_dir, filename)
        
        self.save_tensor_as_image(gen_imgs, filepath)
        # saved_paths.append(filepath)
            
        self.generator.train()
        return gen_imgs

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        
        return checkpoint

# Training setup and execution
def train_emoji_gan(batch_size = 32, num_epochs = 200, image_size = 64, continue_from_last_checkpoint=True):
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 4, [0.5] * 4)  # For RGBA images
    ])
    
    # Create dataset and dataloader
    dataset = EmojiDataset(
        image_dir='./emoji-data/image/JoyPixels',
        captions_file='./emoji-data/emoji_with_captions.json',
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Initialize GAN
    gan = EmojiGAN(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if continue_from_last_checkpoint:
        try:
            print("Trying to load last check point if exists for weight initialization")
            gan.load_checkpoint('emoji_gan_checkpoint/latest_checkpoint.pth')
        except Exception as e:
            print(e)
    
    # Training loop
    model = None
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            results = gan.train_step(batch)
            
            if i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{num_epochs}] "
                    f"[Batch {i}/{len(dataloader)}] "
                    f"[D loss: {results['d_loss']:.4f}] "
                    f"[G loss: {results['g_loss']:.4f}]"
                )
                
        # Save sample images
        if (epoch + 1) % 10 == 0:
            sample_captions = [
                "happy face with tears",
                "angry red face",
                "sleepy face",
                "yawning face"
            ]
            
            for caption in sample_captions:
                gen_imgs = gan.generate_emoji(caption, epoch, num_samples=4)
                # Save generated images...
                
        # Save model checkpoints
        if (epoch + 1) % 50 == 0:
            try:
                generator_state_dict = gan.generator.state_dict()
                discriminator_state_dict = gan.discriminator.state_dict()
                text_encoder_state_dict = gan.text_encoder.state_dict()
                model = {
                    'generator_state_dict': generator_state_dict,
                    'discriminator_state_dict': discriminator_state_dict,
                    'text_encoder_state_dict': text_encoder_state_dict,
                }
                print("Saving model...")
                torch.save(model, f'emoji_gan_checkpoint/emoji_gan_temp.pth')
            except Exception as e:
                print(e)

        # Save latest checkpoint for reference
        if (epoch + 1) % 100 == 0:
            try:
                if model:
                    torch.save(model, f'emoji_gan_checkpoint/latest_checkpoint.pth')
            except Exception as e:
                print(e)


    
        

def generate_custom_emojis():
    # Initialize GAN and load checkpoint
    gan = EmojiGAN()
    gan.load_checkpoint('emoji_gan_checkpoint/latest_checkpoint.pth')
    
    # Generate emojis with custom captions
    captions = [
        "happy face with sunglasses",
        "angry red face",
        "sleepy face",
        "yawning face"
    ]
    for caption in captions:
        # Generate multiple versions
        saved_paths = gan.generate_emoji(
            caption,
            0,
            num_samples=4,
            save_dir="emoji-gan-modified/generated"
        )
        
        print(f"Generated emojis for '{caption}':")
        for path in saved_paths:
            print(f"  - {path}")

if __name__ == "__main__":
    train_emoji_gan(batch_size = 32, num_epochs = 500, image_size = 64)
    generate_custom_emojis()

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.dataset import get_dataloader
from src.model import BaseNetwork, SiameseNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_siamese_network(siamese_network, train_loader, val_loader, num_epochs=10):
    criterion = nn.MSELoss()  # Use Mean Squared Error Loss
    optimizer = optim.Adam(siamese_network.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    writer = SummaryWriter(log_dir='./runs/siamese_network')

    siamese_network.to(device)  # Move model to GPU

    for epoch in range(num_epochs):
        siamese_network.train()
        train_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))  # Initialize tqdm progress bar
        for i, data in progress_bar:
            img1_t1, img2_t1, img1_t2, img2_t2, label = data

            # Move data to GPU
            img1_t1, img2_t1, img1_t2, img2_t2, label = img1_t1.to(device), img2_t1.to(device), img1_t2.to(device), img2_t2.to(device), label.to(device).float()

            optimizer.zero_grad()
            output = siamese_network(img1_t1, img2_t1, img1_t2, img2_t2)
            loss = criterion(output, label.unsqueeze(1))  # Ensure label has the same shape as output
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

            # Update progress bar description
            progress_bar.set_description(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / (i + 1):.4f}')

        train_loss /= len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')

        val_loss = validate(siamese_network, val_loader, criterion)
        writer.add_scalar('Loss/val', val_loss, epoch)

        # Save the model after each epoch
        torch.save(siamese_network.state_dict(), f'siamese_network_epoch_{epoch+1}.pt')

        # Step the scheduler
        scheduler.step()

    writer.close()

def validate(siamese_network, val_loader, criterion):
    siamese_network.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            img1_t1, img2_t1, img1_t2, img2_t2, label = data

            # Move data to GPU
            img1_t1, img2_t1, img1_t2, img2_t2, label = img1_t1.to(device), img2_t1.to(device), img1_t2.to(device), img2_t2.to(device), label.to(device).float()

            output = siamese_network(img1_t1, img2_t1, img1_t2, img2_t2)
            loss = criterion(output, label.unsqueeze(1))  # Ensure label has the same shape as output
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Validation Loss: {val_loss:.4f}')
    return val_loss

if __name__ == '__main__':
    train_csv_path = '/home/r948e/E132-Projekte/Projects/2024_MICCAI_Mario_Challenge/data/Task1/df_task1_train_challenge.csv'
    val_csv_path = '/home/r948e/E132-Projekte/Projects/2024_MICCAI_Mario_Challenge/data/Task1/df_task1_val_challenge.csv'
    train_dir = '/home/r948e/E132-Projekte/Projects/2024_MICCAI_Mario_Challenge/data/Task1/train'
    val_dir = '/home/r948e/E132-Projekte/Projects/2024_MICCAI_Mario_Challenge/data/Task1/val'

    train_loader = get_dataloader(train_csv_path, train_dir)
    val_loader = get_dataloader(val_csv_path, val_dir, shuffle=False)

    base_network = BaseNetwork()
    siamese_network = SiameseNetwork(base_network)

    # Train the Siamese Network
    train_siamese_network(siamese_network, train_loader, val_loader, num_epochs=10)

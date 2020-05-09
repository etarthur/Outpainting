import pickle
import torch
import torchvision
from torch import nn, optim
import matplotlib.pyplot as plt

import outpainting
import multiple_res_model
import residual_model

'''
Edit the paths here and run to train the GAN.
Uses GPU:0 with CUDA (feel free to switch to CPU or use DataParallel).
'''

if __name__ == '__main__':
    print("PyTorch version: ", torch.__version__)
    print("Torchvision version: ", torchvision.__version__)

    # Define paths
    model_save_path = 'outpaint_models'
    html_save_path = 'outpaint_html'
    train_dir = '../train'
    val_dir = '../val'
    test_dir = '../test'

    # Define datasets & transforms
    my_tf = torchvision.transforms.Compose([
            torchvision.transforms.Resize(outpainting.output_size),
            torchvision.transforms.CenterCrop(outpainting.output_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()])
    batch_size = 4
    train_data = outpainting.CEImageDataset(train_dir, my_tf, outpainting.output_size, outpainting.input_size, outpaint=True)
    val_data = outpainting.CEImageDataset(val_dir, my_tf, outpainting.output_size, outpainting.input_size, outpaint=True)
    test_data = outpainting.CEImageDataset(test_dir, my_tf, outpainting.output_size, outpainting.input_size, outpaint=True)
    train_loader = outpainting.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = outpainting.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = outpainting.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)
    print('train:', len(train_data), 'val:', len(val_data), 'test:', len(test_data))

    # Define model & device
    device = torch.device('cuda:0')
    G_net = multiple_res_model.CompletionNetwork()
    CD_net = outpainting.ContextDiscriminator((3, outpainting.output_size, outpainting.output_size), (3, outpainting.output_size, outpainting.output_size), arc='places2')
    G_net.load_state_dict(torch.load("outpaint_models/G_9.pt"))
    CD_net.load_state_dict(torch.load("outpaint_models/D_9.pt"))
    G_net.to(device)
    CD_net.to(device)
    print('device:', device)

    # Start training
    data_loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}  # NOTE: test is evidently not used by the train method
    n_epochs = 50
    adv_weight = [ 0.005, 0.015, 0.040]  # corresponds to epochs 1-10, 10-30, 30-60, 60-onwards
    hist_loss = outpainting.train(G_net, CD_net, device,
                                  criterion_pxl=nn.L1Loss().to(device),
                                  criterion_D=nn.MSELoss().to(device),
                                  optimizer_G=optim.Adam(G_net.parameters(), lr=3e-4, betas=(0.5, 0.999)),
                                  optimizer_D=optim.Adam(CD_net.parameters(), lr=3e-4, betas=(0.5, 0.999)),
                                  data_loaders=data_loaders,
                                  model_save_path=model_save_path,
                                  html_save_path=html_save_path,
                                  n_epochs=n_epochs,
                                  adv_weight=adv_weight)

    # Save loss history and final generator
    pickle.dump(hist_loss, open('hist_loss.p', 'wb'))

    plt.plot(hist_loss['train_pxl'])
    plt.xlabel('epochs')
    plt.ylabel('L1 loss')
    plt.savefig('train_pxl.png')

    plt.plot(hist_loss['train_adv'])
    plt.xlabel('epochs')
    plt.ylabel('MSE loss')
    plt.savefig('train_adv.png')

    plt.plot(hist_loss['train_D'])
    plt.xlabel('epochs')
    plt.ylabel('context discriminator loss')
    plt.savefig('train_D.png')

    plt.plot(hist_loss['val_pxl'])
    plt.xlabel('epochs')
    plt.ylabel('L1 loss')
    plt.savefig('val_pxl.png')

    plt.plot(hist_loss['val_adv'])
    plt.xlabel('epochs')
    plt.ylabel('MSE loss')
    plt.savefig('val_adv.png')

    plt.plot(hist_loss['val_D'])
    plt.xlabel('epochs')
    plt.ylabel('context discriminator loss')
    plt.savefig('val_D.png')

    torch.save(G_net.state_dict(), 'generator_final.pt')

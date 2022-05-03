import argparse
import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vqvae import VQVAE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=str, default='colors')
    parser.add_argument("--checkpoint", type=str, default='vqvae_200.pt')
    parser.add_argument("--batch_size", type=int, default=8)
    
    args = parser.parse_args()
    
    # Set up transforms for dataloader
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    
    # Check if checkpoint file is valid
    if args.checkpoint not in os.listdir('./'):
        raise SystemExit('Checkpoint file not found')
    
    # Instantiate model
    model = VQVAE()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    
    # Set up image paths and list of mean latent variables for each class
    if args.classes == 'colors':
        path = 'cards/'
        cl_means_t = [None, None, None, None, None]
        cl_means_b = [None, None, None, None, None]
    elif args.classes == 'refined':
        path = 'cards_refined/'
        cl_means_t = [None, None]
        cl_means_b = [None, None]
    else:
        raise SystemExit('Invalid class argument. Use \'colors\' or \'refined\'')

    #Set up dataset
    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size // 1, num_workers=0
    )
    
    # Encode images from each class, take running averages of latent variables
    with torch.no_grad():
        print('Encoding images...')
    
        for i, (img, label) in enumerate(loader):
            print('Iteration: ', (i+1), '/', len(iter(loader)))
    
            quant_t, quant_b, diff, _, _ = model.encode(img)
    
            mean_t = torch.mean(quant_t, 0, True)
            mean_b = torch.mean(quant_b, 0, True)
            
            run_means_t = [None]*len(cl_means_t)
            run_means_b = [None]*len(cl_means_b)
            num_labels = [0]*len(cl_means_t)
            
            # Add encoded latent tensors for each image in each class
            for j in range(len(label)):
                num_labels[label[j]] += 1
                if run_means_t[label[j]] is None:
                    run_means_t[label[j]] = quant_t[j]
                else:
                    run_means_t[label[j]] += quant_t[j]
                    
                if run_means_b[label[j]] is None:
                    run_means_b[label[j]] = quant_b[j]
                else:
                    run_means_b[label[j]] += quant_b[j]
            
            # Divide each sum of latent tensors by the number of times it
            # appears. Take the average of the running mean of the current batch
            # with the overall running means.
            for k in range(len(num_labels)):
                if num_labels[k] > 0:
                    run_means_t[k] /= num_labels[k]
                    run_means_b[k] /= num_labels[k]
                    
                    if cl_means_t[k] is None:
                        cl_means_t[k] = run_means_t[k]
                    else:
                        cl_means_t[k] = (cl_means_t[k] + run_means_t[k]) / 2
                        
                    if cl_means_b[k] is None:
                        cl_means_b[k] = run_means_b[k]
                    else:
                        cl_means_b[k] = (cl_means_b[k] + run_means_b[k]) / 2
            
    # Assign names for each class and save the latent tensors
    if args.classes == 'colors':
        classes = ['black','blue','green','red','white']
    elif args.classes == 'refined':
        classes = ['humans','firststrike']
    
    # SAVING CURRENTLY DISABLED TO AVOID OVERWRITING WORKING FILES
    '''
    for cl in range(len(classes)):
        torch.save(cl_means_t[cl], 'means/mean_t_'+classes[cl]+'.pt')
        torch.save(cl_means_b[cl], 'means/mean_b_'+classes[cl]+'.pt')
    '''




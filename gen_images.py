import argparse
import os
from PIL import Image
from random import randrange

import torch
from torchvision import transforms, utils
from vqvae import VQVAE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='inter')
    parser.add_argument("--checkpoint", type=str, default='vqvae_200.pt')
    parser.add_argument("--input_img", type=str, default='random')
    parser.add_argument("--transform_class", type=str, default='firststrike')
    parser.add_argument("--minmax", type=int, default=2000)
    parser.add_argument("--scale", type=float, default=10)
    
    args = parser.parse_args()
    
    # Check if checkpoint file is valid
    if args.checkpoint not in os.listdir('./'):
        raise SystemExit('Checkpoint file not found')
        
    # Check if mode is valid
    if args.mode != 'inter' and args.mode != 'minmax':
        raise SystemExit('Invalid --mode. Use \'inter\' or \'minmax\'')
    
    
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    
    # Load image to transform
    if args.input_img != 'random': 
        img1 = Image.open(args.input_img)
        img1 = transform(img1)
        img1 = torch.reshape(img1, (1, 3, 256, 256))
        
    else:
        classes = os.listdir('cards/')
        class_idx = randrange(len(classes))
        card_idx = randrange(len(os.listdir('cards/'+classes[class_idx])))       
        img_name = os.listdir('cards/'+classes[class_idx])[card_idx]
        
        img1 = Image.open('cards/'+classes[class_idx]+'/'+img_name)
        img1 = transform(img1)
        img1 = torch.reshape(img1, (1, 3, 256, 256))
        
    # Load model
    model = VQVAE()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    
    # Load latent means for image transformation
    cl_means_t = [None, None, None, None, None, None, None, None]
    cl_means_b = [None, None, None, None, None, None, None, None]
    classes = ['humans','firststrike','black','blue','green','red','white']
    
    if args.transform_class not in classes:
        raise SystemExit('--transform_class not in list of classes.')
    for cl in range(len(classes)):
        cl_means_t[cl] = torch.load('means/mean_t_'+classes[cl]+'.pt')
        cl_means_b[cl] = torch.load('means/mean_b_'+classes[cl]+'.pt')
        
    mean_idx = classes.index(args.transform_class)

    t = torch.reshape(cl_means_t[mean_idx], (1, 64, 32, 32))
    b = torch.reshape(cl_means_b[mean_idx], (1, 64, 64, 64))
    
    img_t, img_b, _, _, _ = model.encode(img1)
    
    # Transform using minimum and maximum indices
    if args.mode == 'minmax':
        t, b = t.flatten(), b.flatten()
        img_t, img_b = img_t.flatten(), img_b.flatten()
        
        t_shape, b_shape = t.shape, b.shape
        
        minmax = args.minmax
        max_idx_t = torch.topk(t, minmax).indices
        min_idx_t = torch.topk(-t, minmax).indices
        
        max_idx_b = torch.topk(b, minmax).indices
        min_idx_b = torch.topk(-b, minmax).indices
        
        scale = args.scale
        for idx in range(len(max_idx_t)):
            img_t[max_idx_t[idx]] *= scale
            img_t[min_idx_t[idx]] *= scale
            
        for idx in range(len(max_idx_b)):
            img_b[max_idx_b[idx]] *= scale
            img_b[min_idx_b[idx]] *= scale
            
        img_t = torch.reshape(img_t, (1, 64, 32, 32))
        img_b = torch.reshape(img_b, (1, 64, 64, 64))
        
        
        dec = model.decode(img_t, img_b)
        
    elif args.mode == 'inter':
        
        img_t += t*(scale/2)
        img_b += b*(scale/2)
        
        dec = model.decode(img_t, img_b)
    
    utils.save_image(
        torch.cat([img1, dec], 0),
        'sample/'+img_name[:-4]+'_to_'+args.transform_class+'.png',
        nrow=8,
        normalize=True,
        range=(-1, 1),
    )

# mtg_artgen
ECE695 Final Project

NOTE: Github did not allow me to upload the entire training dataset of cards, I was only able to upload a small subset for testing purposes. All of the saved checkpoint files and saved latent mean tensors in the 'means/' folder were obtained with the full dataset.

VAE was trained using the autoencoder from the following GitHub repository: https://github.com/rosinality/vq-vae-2-pytorch

All of my attempts to create my own autoencoder or GAN failed, I had to use the previous autoecoder just to get something to work with.

MY WORK:

The script 'get_means.py' is used to find the mean latent variables for each class. Note that the code to save the means has been commented out to avoid overwriting the means in the repository that were obtained using the full dataset.

The code is run with the following parameters:

python get_means.py --classes --checkpoint --batch_size

--classes:
  'colors' (default):
    Gets the mean latent variables for each color category of cards in the dataset (black, blue, green, red, white)
    
  'refined':
    Gets the mean latent variables for the refined subsets in the 'cards_refined/' folder (humans, firststrike)
    
--checkpoint:
  Which checkpoint of the model to use. Several are included in the repo, of the form 'vqvae_x.pt' where the x is the number of epochs run during training
  
--batch_size:
  The batch size to use when encoding the images
  
EXAMPLE CALL: python get_means.py --classes=refined --checkpoint=vqvae_200.pt --batch_size=16


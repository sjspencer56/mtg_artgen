# mtg_artgen
ECE695 Final Project

NOTE: Github did not allow me to upload the entire training dataset of cards, I was only able to upload a small subset for testing purposes. All of the saved checkpoint files and saved latent mean tensors in the 'means/' folder were obtained with the full dataset.

VAE was trained using the autoencoder from the following GitHub repository: https://github.com/rosinality/vq-vae-2-pytorch

All of my attempts to create my own autoencoder or GAN failed, I had to use the previous autoecoder just to get something to work with.

MY WORK:

The script 'get_means.py' is used to find the mean latent variables for each class. Note that the code to save the means has been commented out to avoid overwriting the means in the repository that were obtained using the full dataset.

The script is run with the following parameters:

get_means.py --classes --checkpoint --batch_size

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







The script 'gen_images.py' is used to generate trasformed images given an input images, and the class of image it should be transformed into.

The script is run with the following parameters:

gen_images.py --checkpoint --input_img --transform_class --mode --minmax --scale

--checkpoint:

 Checkpoint of the model to use for decoding the latent variables. Same options as in the previous script
 
--input_image:
  
 Path to image in the repository to be transformed. If left blank, will select a random image from the 'cards/' folder. Both input and output image will be   concatenated and saved together
 
--transform_class:

  The class of image that the input image will be transformed into. Options include the color categories 'black', 'blue', 'green', 'red', and 'white', and the refined categories 'humans' and 'firststrike'
 
--mode:

  'inter'
  
  Will linearly interpolate all latent variables in the encoded image to the means of the chosen transform_class
  
  'minmax'
  
  Will interpolate only the latent variables with the minimum and maximum values in the mean of the chosen transform class. The number of these variables is specified in the next parameter
  
--minmax:

  The number of latent variables to use in the minmax mode. For example, using --minmax=1000 will choose the indices of the 1000 latent variables with the greatest values in the mean, and the indices of the 1000 minimum values in the mean
  
  Experimentally, choosing values between 1000 and 5000 seem to product the best results.
  
--scale:

  The amount the latent variables should be changed. Changes start to become apparent with a scale value of around 5, and output starts to become extremely noisy with a scale above 15
  
EXAMPLE CALL: python gen_images.py --checkpoint=vqvae_200.pt --transform_class=red --mode=minmax --scale=7.5 --minmax=1500
  

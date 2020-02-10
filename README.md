# OutliersGAN #

This code implements a one-dimensional Generative Adversarial Network which performs the following tasks:

* It generates realistic 1D data distributions after training. For the case of my project, the data consisted of SDSS spectra;

* It uses the trained model to generate an arbitrarily large number of fake spectra;

* It access the layers of the discriminator after training in order to show that it is possible to perform outlier detection with GANs.

### Notes ###
The following models were trained and are currently saved in OzStar:

- legacy data; extra loss term; grid; valid: 391672                                                                                                                                                                                                                                    
  name: wgangp_grid_newloss 
  code: 71

- galaxy with STARFORMING and STARBURST plus zWarning=0; extra loss term; grid; valid: 321520                                                                                                                                                                                          #   name: gal_zWarning 
  code: 72

- qso plus zWarning=0; extra loss term; grid; valid: 22791                                                                                                                                                                                                                             
  name: qso_zWarning                                                                                                                                                                                                                                                                   co  code 73 
     
- qso plus zWarning=0; extra loss term; grid; valid: 2346
  different wavelength range: 1800-4150 Angstroms                                                                                                                                                                                                                                      
  name: qso2_zWarning   
  code: 74

### Acknowledgements ###
I thank my supervisor, Dr. Karl Glazebrook, for having given both the opportunity and funding to work in such an interesting project during 5 months.
I further thank Dr. Colin Jacobs for being so nice and helpful during my stay.
I finally thank Swinburne University of Technology, and particularly the Centre for Astrophysics and Supercomputing (Melbourne, Australia) for welcoming me extremely well.

from classes import GAN

gan = GAN(latent_dims=5, gen_activation='sigmoid', lr=0.0001, device='gpu:3', labels='soft', generator_type='rnn',
          cadence=None, data_type='sim', z_lim=0.08)
gan.gen_lightcurves(length=15)

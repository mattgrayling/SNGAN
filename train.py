from classes import GAN

gan = GAN(latent_dims=5, gen_activation='sigmoid', lr=0.0001, device='gpu:0', labels='soft', generator_type='rnn',
          cadence=None, data_type='sim', z_lim=0.08, error=True, units=40)
gan.train(epochs=500, plot_interval=1)

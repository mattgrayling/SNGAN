from wgan import WGAN

gan = WGAN(latent_dims=10, device='gpu:2', data_type='sim', z_lim=0.08, mode='observed', gen_units=10, crit_units=5,
           batch_norm=True)
gan.train(epochs=2000, plot_interval=1)

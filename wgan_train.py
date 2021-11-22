from wgan import WGAN

gan = WGAN(latent_dims=10, device='gpu:2', data_type='sim', z_lim=0.08, mode='observed', gen_units=50, crit_units=6,
           batch_norm=False, sn_type='Ic')
# gan.plot_train_sample()
gan.train(epochs=2000, plot_interval=1)

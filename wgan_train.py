from wgan import WGAN

gan = WGAN(latent_dims=10, device='gpu:0', data_type='sim', z_lim=0.065, mode='observed', gen_units=50, crit_units=6,
           batch_norm=False, sn_type='II', lr=0.0005)
# gan.plot_train_sample()
gan.train(epochs=100, plot_interval=1)

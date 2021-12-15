from wgan import WGAN

gan = WGAN(latent_dims=10, device='gpu:0', data_type='sim', z_lim=0.08, mode='observed', gen_units=50, crit_units=10,
           batch_norm=False, sn_type='II', c_dropout=0.7, g_dropout=0.5, clr=0.0005, glr=0.0005, ds=2)
# gan.plot_train_sample()
gan.train(epochs=5000, plot_interval=1)

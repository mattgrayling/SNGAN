from wgan import WGAN

gan = WGAN(latent_dims=10, device='gpu:0', z_lim=0.08, mode='observed', gen_units=1000, crit_units=100,
           batch_norm=False, sn_type='II', c_dropout=0.7, g_dropout=0.5, clr=0.00001, glr=0.00001, ds=1)
# gan.plot_train_sample()
gan.train(epochs=5000, plot_interval=1)

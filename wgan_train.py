from wgan import WGAN

gan = WGAN(latent_dims=10, device='gpu:1', z_lim=0.08, mode='observed', gen_units=50, crit_units=12,
           batch_norm=False, sn_type='II', c_dropout=0.5, g_dropout=0.5, clr=0.0002, glr=0.0002, ds=1,
           inc_colour=True, experiment='wgan-gp', n_critic=3)
# gan.plot_train_sample()
gan.train(epochs=5000, plot_interval=1)

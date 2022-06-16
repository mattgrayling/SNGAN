from wgan import WGAN

gan = WGAN(latent_dims=10, device='gpu:0', z_lim=0.25, mode='observed', gen_units=100, crit_units=100,
           batch_norm=False, sn_type='Ic', c_dropout=0.25, g_dropout=0.25, clr=0.0002, glr=0.0002, ds=1,
           inc_colour=False, experiment='small', n_critic=3, gp_weight=10)
# gan.plot_train_sample()
gan.train(epochs=20000, plot_interval=20000)

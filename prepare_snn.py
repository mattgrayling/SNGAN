from wgan import WGAN

gan = WGAN(latent_dims=10, device='gpu:0', z_lim=0.1, mode='observed', gen_units=100, crit_units=100,
           batch_norm=False, sn_type='II', c_dropout=0.25, g_dropout=0.25, clr=0.00002, glr=0.00002, ds=1,
           inc_colour=False, experiment='tanhscaleerr', n_critic=3, gp_weight=10)
gan.prepare_snn(N=100, epoch=1950)
from wgan import WGAN

gan = WGAN(latent_dims=10, device='gpu:0', z_lim=0.08, mode='observed', gen_units=50, crit_units=50,
           batch_norm=False, sn_type='Ic', c_dropout=0.25, g_dropout=0.25, clr=0.0002, glr=0.0002, ds=1,
           inc_colour=False, experiment='small', n_critic=3, gp_weight=10)
gan.sample_analysis(epoch=3601, n=1, plot_lcs=False, name_suffix=None, file_format='png')
# gan.lc_plot(3, 4, epoch=3601, timesteps=12)

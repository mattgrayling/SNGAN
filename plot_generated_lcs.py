from wgan import WGAN

gan = WGAN(latent_dims=10, device='gpu:0', z_lim=0.15, mode='observed', gen_units=100, crit_units=100,
           batch_norm=False, sn_type='IIb', c_dropout=0.25, g_dropout=0.25, clr=0.00002, glr=0.00002, ds=1,
           inc_colour=False, experiment='tanhscaleerr', n_critic=3, gp_weight=10)
# gan.sample_analysis(epoch=4400, n=1, plot_lcs=False, name_suffix=None, file_format='pdf')
gan.lc_plot(3, 4, epoch=4120, timesteps=12, file_format='pdf')
# gan.prdc(epoch=4120, n=1, repeats=10)
# gan.prdc_explore(start=6700, stop=7200, step=20)

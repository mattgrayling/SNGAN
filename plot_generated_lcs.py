from wgan import WGAN

gan = WGAN(latent_dims=10, z_lim=0.25, gen_units=100, crit_units=100,
           sn_type='Ic', c_dropout=0.25, g_dropout=0.25, clr=0.00002, glr=0.00002,
           experiment='redshift', n_critic=3, gp_weight=10, redshift=True)
gan.sample_analysis(epoch=4500, n=1, plot_lcs=False, name_suffix=None, file_format='pdf')
# gan.lc_plot(3, 4, epoch=4200, timesteps=12, file_format='pdf')
# gan.prdc(epoch=4120, n=1, repeats=10)
# gan.prdc_explore(start=6700, stop=7200, step=20)

from wgan import WGAN

gan = WGAN(latent_dims=10, device='gpu:0', z_lim=0.25, mode='observed', gen_units=50, crit_units=50,
           batch_norm=False, sn_type='Ic', c_dropout=0.25, g_dropout=0.25, clr=0.0002, glr=0.0002, ds=1,
           inc_colour=False, experiment='ncritic', n_critic=3, gp_weight=10)
# gan.plot_lightcurves(n=30, epochs=(1700, ), show=True)
gan.sample_analysis(epoch=1284, n=1, plot_lcs=False, name_suffix=None)

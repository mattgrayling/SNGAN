from cgan import CGAN

gan = CGAN(latent_dims=10, gen_activation='sigmoid', dlr=0.0005, glr=0.0005, epsilon=1, beta1=0.9, beta2=0.98,
           device='gpu:1', labels='soft', data_type='sim', z_lim=0.08, mode='observed', units=100, batch_norm=True)
gan.train(epochs=2000, plot_interval=1, skip_gen=1)

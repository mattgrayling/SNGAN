from cgan import CGAN

gan = CGAN(latent_dims=5, gen_activation='sigmoid', dlr=0.0001, glr=0.0001, device='gpu:1', labels='soft',
           data_type='sim', z_lim=0.08, mode='template', units=10)
gan.colour_analysis(n=10)

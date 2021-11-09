class GANTest:
    def __init__(self, latent_dims=100, mode='flux', N=1000, lr=0.00001,
                 gen_activation='sigmoid'):
        self.latent_dims = latent_dims
        self.mode = mode
        self.N = N
        self.lr = lr
        self.gen_activation = gen_activation
        self.root = 'Data/Models'
        self.name = f'{self.mode}_{self.gen_activation}_lr{self.lr}_ld{self.latent_dims}'
        self.generator_path = os.path.join(self.root, f'{self.name}_model_weights.h5')
        # Dataset properties
        self.glob_t0, self.glob_t0_std = 25, 5  # this describes the t0 values for the population
        self.cadence, self.cadence_std = 6, 2  # random light curve sampling
        self.err, self.err_std = 0.2, 0.04
        self.tlim = 150

        # Build dataset
        self.dataset_path = os.path.join(self.root, f'{self.mode}_data.pkl')
        if not os.path.exists(self.dataset_path):
            self.X, self.maxes = self.__build_dataset__()
            pickle.dump([self.X, self.maxes], open(self.dataset_path, 'wb'))
        else:
            self.X, self.maxes = pickle.load(open(self.dataset_path, 'rb'))

        # Optimizer
        self.optimizer = opt.Adam(learning_rate=self.lr)

        # Build discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy')
        print(self.discriminator.summary())

        # Build generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        print(self.generator.summary())

        # Build combined model
        i = Input(shape=(None, self.latent_dims))
        lcs = self.generator(i)

        self.discriminator.trainable = False

        valid = self.discriminator(lcs)

        self.combined = Model(i, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    @staticmethod
    def lc(t, t0):
        return t * np.exp(-t / t0) + 0.025 * t

    def lc_sample(self):
        t = 0
        t0 = rand.normal(self.glob_t0, self.glob_t0_std)
        T, F, E = [], [], []
        while t < self.tlim:
            t += 5  # np.abs(rand.normal(cadence,cadence_std))
            e = np.abs(rand.normal(self.err, self.err_std))
            f = rand.normal(self.lc(t, t0), e)
            T.append(t)
            F.append(f)
            E.append(e)
        data = np.array([T, F]).T  # ,E
        return data

    def scale_down(self, X):
        if self.mode == 'flux' and self.gen_activation == 'tanh':
            maxes = []
            for i in range(X.shape[-1]):
                max = np.max(X[:, :, i])
                X[:, :, i] = X[:, :, i] - (max / 2)
                X[:, :, i] = X[:, :, i] / (max / 2)
                maxes.append(max)
            return X, maxes
        else:
            raise ValueError('Please implement this combination of activation function and mode')

    def scale_up(self, X):
        if self.mode == 'flux' and self.gen_activation == 'tanh':
            for i in range(X.shape[-1]):
                X[:, :, i] = X[:, :, i] * (self.maxes[i] / 2)
                X[:, :, i] = X[:, :, i] + (self.maxes[i] / 2)
            return X

    def __build_dataset__(self):
        X = np.zeros((self.N, 30, 2))

        print('Generating dataset...')
        for i in tqdm(range(self.N)):
            x = self.lc_sample()
            x[:, 1] = x[:, 1] * 1e-15
            if self.mode == 'mag':
                x[:, 1] = -2.5 * np.log10(x[:, 1]) - 18
            X[i, :, :] = x

            plt.scatter(X[i, :, 0], X[i, :, 1])
            # plt.errorbar(X[i,:,0],X[i,:,1],X[i,:,2],fmt='x')
        if self.mode == 'mag':
            plt.gca().invert_yaxis()
        plt.show()

        # Normalise X data
        print('Scaling data...')
        X, maxes = self.scale_down(X)
        '''
        elif self.mode == 'mag':
          max1 = np.max(X[:,:,i])
          X[:,:,i] = (X[:,:,i] - max1) * -1
          max2 = np.max(X[:,:,i])
          X[:,:,i] = X[:,:,i]/max2      
          maxes.append([max1, max2])
        '''
        plt.figure()
        for i in range(self.N):
            plt.scatter(X[i, :, 0], X[i, :, 1])
            # plt.errorbar(X[i,:,0],X[i,:,1],X[i,:,2],fmt='x')
        plt.show()
        print('Done')

        return X, maxes

    def build_generator(self):
        input = Input(shape=(None, self.latent_dims))
        gru1 = GRU(100, return_sequences=True)(input)
        # bn1 = BatchNormalization()(gru1)
        gru2 = GRU(100, return_sequences=True)(gru1)
        # bn2 = BatchNormalization()(gru2)
        output = GRU(2, return_sequences=True, activation=self.gen_activation)(gru2)
        model = Model(input, output)
        return model

    def build_discriminator(self):
        input = Input(shape=(None, 2))
        gru1 = GRU(100, return_sequences=True)(input)
        # bn1 = BatchNormalization()(gru1)
        gru2 = GRU(100, return_sequences=True)(gru1)
        # bn2 = BatchNormalization()(gru2)
        output = GRU(1, activation='sigmoid')(gru2)
        model = Model(input, output)
        return model

    def train(self, epochs=100, batch_size=200, plot_interval=100):
        n_batches = int(self.X.shape[0] / batch_size)
        rng = np.random.default_rng(123)

        if os.path.exists(self.generator_path):
            raise ValueError('A weights path already exists for this model, please delete '
                             'or rename it')

        for epoch in range(epochs):
            rng.shuffle(self.X, axis=0)
            g_losses, d_losses = [], []
            for batch in range(n_batches):
                # real = np.zeros((batch_size,1))
                # fake = np.ones((batch_size,1))
                real = np.random.uniform(0.0, 0.1, (batch_size, 1))
                fake = np.random.uniform(0.9, 1.0, (batch_size, 1))

                '''
                for i in range(int(real.shape[0]/20)):
                  rand_ind = np.random.randint(real.shape[0])
                  real[rand_ind, 0] = np.random.uniform(0.9, 1.0)

                for i in range(int(fake.shape[0]/20)):
                  rand_ind = np.random.randint(fake.shape[0])
                  fake[rand_ind, 0] = np.random.uniform(0.0, 0.1)
                '''

                X_sub = self.X[batch * batch_size:batch * batch_size + batch_size, :, :]

                # Generate fake data
                rand_length = 30  # int(rand.normal(30,3))
                noise = rand.normal(size=(batch_size, rand_length, self.latent_dims))
                gen_lcs = self.generator.predict(noise)

                # Train discriminator
                d_loss_real = self.discriminator.train_on_batch(X_sub, real)
                d_loss_fake = self.discriminator.train_on_batch(gen_lcs, fake)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train generator
                rand_length = 30  # int(rand.normal(30,3))
                # noise = rand.normal(size=(2*batch_size,rand_length,1))
                noise = rand.normal(size=(2 * batch_size, rand_length, self.latent_dims))

                g_loss = self.combined.train_on_batch(noise, np.zeros((2 * batch_size, 1)))

                g_losses.append(g_loss)
                d_losses.append(d_loss)
            self.generator.save_weights(self.generator_path)
            full_g_loss = np.mean(g_loss)
            full_d_loss = np.mean(d_loss)
            print(f'{epoch + 1}/{epochs} g_loss={full_g_loss}, d_loss={full_d_loss}'
                  f' Ranges: x [{np.min(gen_lcs[:, :, 0])}, {np.max(gen_lcs[:, :, 0])}], '
                  f'y [{np.min(gen_lcs[:, :, 1])}, {np.max(gen_lcs[:, :, 1])}]')

            if (epoch + 1) % plot_interval == 0:
                plot_test = gen_lcs[0, :, :]
                if self.mode == 'flux':
                    x = plot_test[:, 0]  # * self.maxes[0] #* stds[0] + means[0]
                    y = plot_test[:, 1]  # * self.maxes[1] #* stds[1] + means[1]
                elif self.mode == 'mag':
                    x = plot_test[:, 0] * self.maxes[0]  # * stds[0] + means[0]
                    y = plot_test[:, 1] * self.maxes[1][1] * -1 + self.maxes[1][0]  # * stds[1] + means[1]
                plt.figure()
                plt.scatter(x, y)
                if self.mode == 'mag':
                    plt.gca().invert_yaxis()
                plt.show()

    def gen_lightcurves(self, n=10, length=30):
        noise = rand.normal(size=(n, length, self.latent_dims))
        self.generator.load_weights(self.generator_path)
        gen_lcs = self.generator.predict(noise)
        gen_lcs = self.scale_up(gen_lcs)
        scaled_X = self.scale_up(self.X.copy())
        for i in range(n):
            # Plot generated data
            data = gen_lcs[i, :, :]
            x = data[:, 0]
            if self.mode == 'mag':
                y = data[:, 1] * self.maxes[1][1] * -1 + self.maxes[1][0]
            else:
                y = data[:, 1]
            plt.scatter(x, y, label='Generated')
            # Plot real data for comparison
            rand_ind = np.random.randint(self.X.shape[0])
            real_data = scaled_X[rand_ind, :, :]
            x = real_data[:, 0]
            if self.mode == 'mag':
                y = real_data[:, 1] * self.maxes[1][1] * -1 + self.maxes[1][0]
            else:
                y = real_data[:, 1]
            plt.scatter(x, y, label='Real')
            if self.mode == 'mag':
                plt.gca().invert_yaxis()
            # Show plot
            plt.legend()
            plt.show()

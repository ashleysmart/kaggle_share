import numpy as np
import keras.preprocessing.image

class DataGenerator:
    PATCHES = 1
    ROI     = 2

    #https://stackoverflow.com/questions/46271896/how-to-use-keras-imagedatagenerator-for-image-transformation-inputs
    def _create_gens(self, data, seed):
        gen_args = dict(
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.1,
            vertical_flip=True,
            horizontal_flip=True,
            fill_mode="constant",
            cval=0.0)

        gens = []

        for i in range(data.shape[3]):
            shape = data.shape[0:3] + (1,)
            select = data[:,:,:,i].reshape(shape)

            gen = keras.preprocessing.image.ImageDataGenerator(**gen_args)
            gen.fit(select, augment=True, seed=seed)

            gens.append(gen)
            
        return gens
    
    # transforms flowing out of the generators.. 
    def _transform(self,
                   gens,
                   data,
                   add_noise=True,
                   seed=1):

        # note on the shape - data is multiple images bt keras will randomly select one and transform it (due to batch_size=1)
        transform_shape = data.shape[1:4]
        transform = np.zeros(transform_shape, dtype=np.float)

        shape = data.shape[0:3] + (1,)

        for i in range(len(gens)):
            # crazyness with ImageDataGenerator.. im using the single channel(greyscale) tansformer
            select = data[:,:,:,i].reshape(shape)

            # note on the batch size of 1.. images are large.. im going to chop them to bits so lets not queue up to much.. 
            # note on the seed.. we must force it so sync all the transforms for all the channels 
            itr = gens[i].flow(select,batch_size=1,seed=seed)
            d = itr.next()

            # almost human invisiable random noise injection
            transform[:,:,i] = d.reshape(transform_shape[0:2])
            if add_noise:
                noise = (np.random.random(transform_shape[0:2]) - 0.5) / 128.0
                transform[:,:,i] += noise
                
        return transform

    # patchs choosen from the transformed image stack
    def _select_patches(self,
                    data_in,   data_out,
                    patches_in,patches_out,
                    offset,    size):
        xmax = data_in.shape[0] - self.patch_size
        ymax = data_in.shape[1] - self.patch_size

        # because of random noise.. 
        patch_threshold = self.patch_size*self.patch_size*self.data_in.shape[3] / 128.0

        # make random patch selections into generated dataset...
        idx = 0
        while idx < size:
            xstart = np.random.randint(xmax)
            ystart = np.random.randint(ymax)

            patches_in [offset+idx] = data_in [xstart:(xstart + self.patch_size), ystart:(ystart + self.patch_size),:]
            patches_out[offset+idx] = data_out[xstart:(xstart + self.patch_size), ystart:(ystart + self.patch_size),:]
            
            # reject the image if there isnt sufficent data in it 
            # ie if it has less than 100 pixels colored in all inputs repeat the selection
            if np.sum(patches_in[offset+idx]) > patch_threshold:
                idx = idx + 1

    # patchs choosen from the transformed image stack
    def _select_roi(self,
                    data_in,   data_out,
                    patches_in,roi_out,
                    offset,    size):
        xmax = data_in.shape[0] - self.patch_size
        ymax = data_in.shape[1] - self.patch_size

        # because of random noise.. 
        patch_threshold = self.patch_size*self.patch_size*self.data_in.shape[3] / 128.0

        # make random patch selections into generated dataset...
        idx = 0
        while idx < size:
            xstart = np.random.randint(xmax)
            ystart = np.random.randint(ymax)

            patches_in[offset+idx] = data_in [xstart:(xstart + self.patch_size), ystart:(ystart + self.patch_size),:]
            mask                   = data_out[xstart:(xstart + self.patch_size), ystart:(ystart + self.patch_size),:]

            raveled_masks = mask.reshape((np.prod(mask.shape[0:-1]),mask.shape[-1]))         
            roi_out[offset+idx] = (np.sum(raveled_masks, axis=0) > 0.0) * 1.0

            # reject the image if there isnt sufficent data in it 
            # ie if it has less than 100 pixels colored in all inputs repeat the selection
            if np.sum(patches_in[offset+idx]) > patch_threshold:
                idx = idx + 1

    def __init__(self, 
                 data_in, data_out,
                 patchs_per_transform = 200,
                 patch_size           = 200,
                 mode                 = PATCHES):
        self.data_in  = data_in
        self.data_out = data_out
        self.mode     = mode

        seed = 1
        self.gen_in  = self._create_gens(data_in,  seed)
        self.gen_out = self._create_gens(data_out, seed)

        self.patchs_per_transform = patchs_per_transform
        self.patch_size           = patch_size

    def __call__(self, epochs, samples):
        patches_in  = np.zeros((samples,self.patch_size,self.patch_size,self.data_in.shape[3]),  dtype=np.float)

        if self.mode == DataGenerator.PATCHES:
            out = np.zeros((samples,self.patch_size,self.patch_size,self.data_out.shape[3]), dtype=np.float)
        else:
            out = np.zeros((samples,self.data_out.shape[3]), dtype=np.float)

        for i in range(epochs):
            seed  = np.random.randint(65000)
            offset = 0
            count  = samples

            while count > 0:
                # transform a new image 
                transform_in  = self._transform(self.gen_in,  self.data_in,  add_noise=True,  seed=(seed + count))
                transform_out = self._transform(self.gen_out, self.data_out, add_noise=False, seed=(seed + count))

                # extract patches from transformed image
                patch_count = min(self.patchs_per_transform, count)
                if self.mode == DataGenerator.PATCHES:
                    self._select_patches(transform_in, transform_out, 
                                         patches_in, out,
                                         offset, patch_count)
                else:
                    self._select_roi(transform_in, transform_out, 
                                         patches_in, out,
                                         offset, patch_count)

                count  = count  - patch_count
                offset = offset + patch_count
            yield patches_in, out
                

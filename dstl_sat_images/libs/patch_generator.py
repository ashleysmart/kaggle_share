import numpy as np
import keras.preprocessing.image

# note transforms operate on a single data point with channels to the right most
class NullTransform:
    def __init__(self):
        pass

    # NOTE call with for multiple items
    def allocate(self,samples,patch_size,channels):
        return np.zeros((samples,patch_size,patch_size,channels),dtype=np.float)

    # NOTE call with single item!
    def __call__(self, data):
        return data

class RoiTransform:
    def __init__(self):
        pass

    # NOTE call with for multiple items
    def allocate(self,samples,patch_size,channels):
        return np.zeros((samples,channels),dtype=np.float)

    # NOTE called with a single item!
    def __call__(self, data):
        raveled = data.reshape((np.prod(data.shape[0:2]), data.shape[2]))
        return (np.sum(raveled, axis=0) > 0.0) * 1.0

class DataGenerator:
    #https://stackoverflow.com/questions/46271896/how-to-use-keras-imagedatagenerator-for-image-transformation-inputs
    def _create_gens(self, channels, seed):
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

        for i in range(channels):
            gen = keras.preprocessing.image.ImageDataGenerator(**gen_args)
            gens.append(gen)
            
        return gens
    
    # augmentation transforms flowing out of the generators.. 
    def _augmentation(self,
                      gens,
                      data,
                      seed=1):
        # note on the shape - data is a single images but the flow selected it from a batch
        # data    is:        height, width, channels 
        # augment is:        height, width, channels 
        # gen     is: batch, height, width, channels 
        gen_shape  = (1,) + data.shape[:2] + (1,)
        chan_shape =        data.shape[:2] 

        augment = np.zeros(data.shape, dtype=np.float)

        for i in range(len(gens)):
            # crazyness with ImageDataGenerator.. im using the single channel(greyscale) augmenter
            select = data[:,:,i].reshape(gen_shape)
            
            # note on the batch size of 1.. images are large.. im going to chop them to bits so lets not queue up to much.. 
            # note on the seed.. we must force it so sync all the transfors for all the channels 
            itr = gens[i].flow(select,batch_size=1,seed=seed)
            d = itr.next()

            augment[:,:,i] = d.reshape(chan_shape)
                
        return augment

    # patchs choosen from the transformed image stack
    def _select(self,
                data_in,      data_out,
                select_in,    select_out,
                offset,       size):
        # note data format: height, width, channels
        
        ymax = data_in.shape[0] - self.patch_size 
        xmax = data_in.shape[1] - self.patch_size

        # make random patch selections into generated dataset...
        idx = 0
        while idx < size:
            ystart = np.random.randint(ymax)
            xstart = np.random.randint(xmax)

            patch_in  = data_in [ystart:(ystart + self.patch_size), xstart:(xstart + self.patch_size), :]
            patch_out = data_out[ystart:(ystart + self.patch_size), xstart:(xstart + self.patch_size), :]

            # reject the image if there isnt sufficent data in it 
            # ie if it has less than 100 pixels colored in all inputs repeat the selection
            if np.sum(patch_in) < 100.0:
                continue

            # almost human invisiable random noise injection on input
            patch_in += (np.random.random(patch_in.shape) - 0.5) / 128.0
            
            select_in [offset+idx] = self.transform_in (patch_in)
            select_out[offset+idx] = self.transform_out(patch_out)

            idx = idx + 1

    def __init__(self, 
                 channels_in, channels_out,
                 patchs_per_augmentation = 200,
                 patch_size              = 200,
                 transform_in            = NullTransform(),
                 transform_out           = NullTransform()):

        self.channels_in  = channels_in
        self.channels_out = channels_out

        seed = 1
        self.gen_in  = self._create_gens(channels_in,  seed)
        self.gen_out = self._create_gens(channels_out, seed)

        self.patchs_per_augmentation = patchs_per_augmentation
        self.patch_size              = patch_size

        self.transform_in  = transform_in 
        self.transform_out = transform_out

    def __call__(self, data_flow, epochs, samples):
        select_in  = self.transform_in .allocate(samples, self.patch_size, self.channels_in )
        select_out = self.transform_out.allocate(samples, self.patch_size, self.channels_out)

        for i in range(epochs):
            seed  = np.random.randint(65000)
            offset = 0
            count  = samples

            while count > 0:
                # a flow so we can use either in mem or in disk data
                data_in, data_out = data_flow.select_random()
                
                # augmentation of a new image 
                # data shape: channels, height, width
                augmentation_in  = self._augmentation(self.gen_in,  data_in,  seed=(seed + count))
                augmentation_out = self._augmentation(self.gen_out, data_out, seed=(seed + count))

                # extract patches from transformed image
                patch_count = min(self.patchs_per_augmentation, count)

                self._select(augmentation_in, augmentation_out, 
                             select_in,       select_out,
                             offset,          patch_count)

                count  = count  - patch_count
                offset = offset + patch_count
            yield select_in, select_out
                

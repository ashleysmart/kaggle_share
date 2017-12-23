import numpy as np
import time

import time
def timer_sync():
    tnow = time.time()
    return tnow
    
def timer_check(idx,tots,prior):
    tnow = time.time()
    tots[idx] += tnow - prior
    return tnow

def timer_finish(tots):
    print "   ....timing:", tots


class PatchInterpolationSequancer:
    # this class is about extracting a sequance of patches from a image, processoring them and merging the results
    # back into the original while interpolating the edges of the patches

    # this allows you to overlap model predictions on patches from an image and blend them together at the edges
    
    # it uses interpolation masks to merge the transformed patches into the final image
    def generate_interploation_vectors(self):    
        # compute patch interpolation mask...
        # linear mask so outer edge should be almost zero.. accounts for 1-4 overlaping pixels
        # internal pixels should still range between [0,1.0] after summnation
        edge_overlap = 2*self.patch_edge

        row_left  = np.ones(self.patch_step + self.patch_edge)
        row_mid   = np.ones(self.patch_size)
        row_right = np.ones(self.patch_edge + self.patch_step)

        # oddies here.. edge doesnt include 0 and 1 (that would be waste computation)
        m1 = 1.0/float(edge_overlap+1)
        x1 = (np.arange(1,edge_overlap+1))*m1

        row_mid  [:edge_overlap] = x1
        row_right[:edge_overlap] = x1

        x1 = np.fliplr([x1])[0]

        row_mid  [(row_mid .shape[0] - edge_overlap):] = x1
        row_left [(row_left.shape[0] - edge_overlap):] = x1
        
        return [row_left, row_mid, row_right]
    
    def generate_masks(self):
        rows = self.generate_interploation_vectors()

        masks = []
        for row in rows:
            mask_row = []
            for col in rows:
                mask = np.matmul(col.reshape((col.shape[0],1)), row.reshape((1,row.shape[0])))
                mask_row.append(mask)
            masks.append(mask_row)
    
        return masks

    def mask_idx(self,idx,limit):
        if idx == 0:
            # left/top edge
            return 0, self.patch_edge, self.patch_size
        elif idx == limit-1:
            # right/bottom
            return 2, 0, self.patch_size - self.patch_edge
        else:
            return 1, 0, self.patch_size 
        
    def access_ranges(self, x, img_size):
        img_min = self.patch_step*x - self.patch_edge
        img_max = img_min + self.patch_size

        patch_min = -min(0,img_min)                        # the part of the patch outside the left edge
        patch_max = self.patch_size - max(0,img_max - img_size) # patch size - the part of the patch out side the right edge

        img_min = max(0,img_min)
        img_max = min(img_size,img_max)
        
        return patch_min, patch_max, img_min, img_max


    def __init__(self,
                patch_size,  
                patch_edge,
                chan_in,
                chan_out): 
        self.patch_size  = patch_size  # size (square) of the patch to extract
        self.patch_edge  = patch_edge  # overlap in the edge of the patches (interpolation area)

        # computed settings
        self.patch_step = self.patch_size - 2*self.patch_edge
        self.patch_off  = int((self.patch_size - self.patch_step)/2)

        # channels
        self.chan_out = chan_out
        self.chan_in = chan_in

        # interpolation masks
        self.masks = self.generate_masks()

    def __call__(self, img, process, shaper=lambda x: x):
        image_shape = img.shape
        
        y_steps = int(np.ceil(float(image_shape[0])/self.patch_step))
        x_steps = int(np.ceil(float(image_shape[1])/self.patch_step))

        out_shape   = image_shape[0:2] +  (self.chan_out,)
        
        patch_shape = (y_steps*x_steps, self.patch_size, self.patch_size, image_shape[2])

        timing = [0 for i in range(6)]

        # prepare patches
        tnow = timer_sync()                
        patch_in = np.zeros(patch_shape)
        for y in range(y_steps):
            for x in range(x_steps):       
                idx = y*x_steps + x
                # compute the in/out image access location
                patch_y_min, patch_y_max, img_y_min, img_y_max = self.access_ranges(y, image_shape[0])
                patch_x_min, patch_x_max, img_x_min, img_x_max = self.access_ranges(x, image_shape[1])

                tnow = timer_check(0,timing, tnow)

                #print "mask:", mask.shape
                #print "x", x, "patch:", patch_x_min, patch_x_max, "img:",img_x_min, img_x_max 
                #print "y", y, "patch:", patch_y_min, patch_y_max, "img:",img_y_min, img_y_max
                
                #select the patch (zeroes for the edges)
                patch_select = patch_in[idx]
                patch_select[patch_y_min:patch_y_max,patch_x_min:patch_x_max,:] = img[img_y_min:img_y_max,img_x_min:img_x_max,:] 

                tnow = timer_check(1,timing, tnow)

        # process the patches
        patch_out = process(patch_in)
        del patch_in

        tnow = timer_check(2,timing,tnow)

        # merge patches
        out = np.zeros(out_shape)
        for y in range(y_steps):
            mask_y, mask_y_min, mask_y_max = self.mask_idx(y,y_steps)
            for x in range(x_steps):
                idx = y*x_steps + x
                mask_x, mask_x_min, mask_x_max = self.mask_idx(x,x_steps)            
                mask = self.masks[mask_x][mask_y]
                
                tnow = timer_check(3,timing,tnow)

                # compute the in/out image access location
                patch_y_min, patch_y_max, img_y_min, img_y_max = self.access_ranges(y, image_shape[0])
                patch_x_min, patch_x_max, img_x_min, img_x_max = self.access_ranges(x, image_shape[1])

                patch_select = shaper(patch_out[idx])
                    
                # mask the patch (interpolation scale its edges)
                for i in range(self.chan_out):
                    patch_select[mask_y_min:mask_y_max,mask_x_min:mask_x_max,i] *= mask                

                tnow = timer_check(4,timing,tnow)

                # merge the patch into the output
                out[img_y_min:img_y_max,img_x_min:img_x_max,:] +=  patch_select[patch_y_min:patch_y_max,patch_x_min:patch_x_max,:]

                tnow = timer_check(5,timing,tnow)

        timer_finish(timing)

        return out

# note this is a channels first implemenation
import numpy as np

class DataFlowFromMemory:
    # this class assumes
    #   the input file is a single datapoint channel first npy
    #   the out   files are a single datapont signel channel npy
    def __init__(self,
                 idents,
                 channels,
                 input_pattern,
                 output_pattern):
        # load up inputs
        ins  = None
        for i in range(len(idents)):
            ident = idents[i]

            inp = np.load(input_pattern(ident))
            if type(inp) == np.lib.npyio.NpzFile:
               inp = inp['arr_0'] 
            inp = inp.astype(np.float)/255.0

            if ins is None:
                shape = (len(idents),) + inp.shape
                ins = np.zeros(shape)
            ins[i] = inp
        self.ins = ins

        out_shape = self.ins.shape[:3] + (len(self.channels),)
        out = np.zeros(out_shape)
        for i in range(len(idents)):
            ident = idents[i]
            for j in range(len(channels)):
                chan = channels[j]
                outc = np.load(output_pattern(ident,chan))
                if type(outc) == np.lib.npyio.NpzFile:
                   outc = outc['arr_0'] 
        
                out[i,:,:,j] = outc * 1.0
        self.out = out

    def select_random(self):
        # randomly select an entry
        select = np.random.randint(self.out.shape[0])
        return self.ins[select], self.out[select]

class DataFlowFromDisk:
    # this class assumes
    #   the input file is a single datapoint channel first npy
    #   the out   files are a single datapont signel channel npy
    def __init__(self,
                 idents,
                 channels,
                 input_pattern,
                 output_pattern):
        self.idents         = idents
        self.channels       = channels
        self.input_pattern  = input_pattern
        self.output_pattern = output_pattern
        
    def select_random(self):
        # randomly select an entry
        select = np.random.randint(len(self.idents))
        ident = self.idents[select]

        # load that entry (uint8 0-255
        ins = np.load(self.input_pattern(ident))
        if type(ins) == np.lib.npyio.NpzFile:
           ins = ins['arr_0'] 
        ins = ins.astype(np.float)/255.0

        out_shape = ins.shape[:2] + (len(self.channels),)

        out = np.zeros(out_shape)
        for j in range(len(self.channels)):
            chan = self.channels[j]
            outc = np.load(self.output_pattern(ident,chan))
            if type(outc) == np.lib.npyio.NpzFile:
               outc = outc['arr_0'] 
            out[:,:,j] = outc * 1.0

        return ins, out
    

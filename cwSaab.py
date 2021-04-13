# v 2021.04.12
# A generalized version of channel wise Saab
# modified from https://github.com/ChengyaoWang/PixelHop-_c-wSaab/blob/master/cwSaab.py
# Note: Depth goal may not achieved if no nodes's energy is larger than energy threshold or too few SaabArgs/shrinkArgs, (warning generates)

import numpy as np 
import gc, time
from saab import Saab

# Python Virtual Machine's Garbage Collector
def gc_invoker(func):
    def wrapper(*args, **kw):
        value = func(*args, **kw)
        gc.collect()
        time.sleep(0.5)
        return value
    return wrapper

class cwSaab():
    def __init__(self, depth=1, TH1=0.01, TH2=0.005, SaabArgs=None, shrinkArgs=None, load=False):
        self.trained = False
        self.split = False
        
        if load == False:
            assert (depth > 0), "'depth' must > 0!"
            assert (SaabArgs != None), "Need parameter 'SaabArgs'!"
            assert (shrinkArgs != None), "Need parameter 'shrinkArgs'!"
            self.depth = (int)(depth)
            self.shrinkArgs = shrinkArgs
            self.SaabArgs = SaabArgs
            self.par = {}
            self.bias = {}
            self.TH1 = TH1
            self.TH2 = TH2
            self.Energy = {}
        
            if depth > np.min([len(SaabArgs), len(shrinkArgs)]):
                self.depth = np.min([len(SaabArgs), len(shrinkArgs)])
                print("       <WARNING> Too few 'SaabArgs/shrinkArgs' to get depth %s, actual depth: %s"%(str(depth),str(self.depth)))

    @gc_invoker
    def SaabTransform(self, X, saab, layer, train=False):
        '''
        Get saab features. 
        If train==True, remove leaf nodes using TH1, only leave the intermediate node's response
        '''
        shrinkArg, SaabArg = self.shrinkArgs[layer], self.SaabArgs[layer]
        assert ('func' in shrinkArg.keys()), "shrinkArg must contain key 'func'!"
        X = shrinkArg['func'](X, shrinkArg)
        S = list(X.shape)
        X = X.reshape(-1, S[-1])
        
        if SaabArg['num_AC_kernels'] != -1:
            S[-1] = SaabArg['num_AC_kernels']
            
        transformed = saab.transform(X)
        transformed = transformed.reshape(S[0],S[1],S[2],-1)
        
        if train==True and self.SaabArgs[layer]['cw'] == True: # remove leaf nodes
            transformed = transformed[:, :, :, saab.Energy>self.TH1]
            
        return transformed
    
    @gc_invoker
    def SaabFit(self, X, layer, bias=0):
        '''Learn a saab model'''
        shrinkArg, SaabArg = self.shrinkArgs[layer], self.SaabArgs[layer]
        assert ('func' in shrinkArg.keys()), "shrinkArg must contain key 'func'!"
        X = shrinkArg['func'](X, shrinkArg)
        S = list(X.shape)
        X = X.reshape(-1, S[-1])
        saab = Saab(num_kernels=SaabArg['num_AC_kernels'], needBias=SaabArg['needBias'], bias=bias)
        saab.fit(X)
        return saab

    @gc_invoker
    def discard_nodes(self, saab):
        '''Remove discarded nodes (<TH2) from the model'''
        energy_k = saab.Energy
        discard_idx = np.argwhere(energy_k<self.TH2)
        saab.Kernels = np.delete(saab.Kernels, discard_idx, axis=0) 
        saab.Energy = np.delete(saab.Energy, discard_idx)
        saab.num_kernels -= discard_idx.size
        return saab

    @gc_invoker
    def cwSaab_1_layer(self, X, train, bias=None):
        '''cwsaab/saab transform starting for Hop-1'''
        if train == True:
            saab_cur = []
            bias_cur = []
        else:
            saab_cur = self.par['Layer'+str(0)]
            bias_cur = self.bias['Layer'+str(0)]
        transformed, eng = [], []
        
        if self.SaabArgs[0]['cw'] == True: # channel-wise saab
            S = list(X.shape)
            S[-1] = 1
            X = np.moveaxis(X, -1, 0)
            for i in range(X.shape[0]):
                X_tmp = X[i].reshape(S)
                if train == True:
                    # fit
                    saab = self.SaabFit(X_tmp, layer=0)
                    # remove discarded nodes
                    saab = self.discard_nodes(saab)
                    # store
                    saab_cur.append(saab)
                    bias_cur.append(saab.Bias_current)
                    eng.append(saab.Energy)
                    # transformed feature
                    transformed.append(self.SaabTransform(X_tmp, saab=saab, layer=0, train=True))
                else:
                    if len(saab_cur) == i:
                        break
                    transformed.append(self.SaabTransform(X_tmp, saab=saab_cur[i], layer=0))
            transformed = np.concatenate(transformed, axis=-1)
        else: # saab, not channel-wise
            if train == True:
                saab = self.SaabFit(X, layer=0)
                saab = self.discard_nodes(saab)
                saab_cur.append(saab)
                bias_cur.append(saab.Bias_current)
                eng.append(saab.Energy)
                transformed = self.SaabTransform(X, saab=saab, layer=0, train=True)
            else:
                transformed = self.SaabTransform(X, saab=saab_cur[0], layer=0)
                
        if train == True:
            self.par['Layer0'] = saab_cur
            self.bias['Layer'+str(0)] = bias_cur
            self.Energy['Layer0'] = eng
                
        return transformed

    @gc_invoker
    def cwSaab_n_layer(self, X, train, layer):
        '''cwsaab/saab transform starting from Hop-2'''
        output, eng_cur, ct, pidx = [], [], -1, 0
        S = list(X.shape)
        saab_prev = self.par['Layer'+str(layer-1)]
        bias_prev = self.bias['Layer'+str(layer-1)]

        if train == True:
            saab_cur = []
            bias_cur = []
        else:
            saab_cur = self.par['Layer'+str(layer)]
        
        if self.SaabArgs[layer]['cw'] == True: # channel-wise saab
            S[-1] = 1
            X = np.moveaxis(X, -1, 0)
            for i in range(len(saab_prev)):
                for j in range(saab_prev[i].Energy.shape[0]):
                    if train==False:
                        ct += 1 # helping index
                    if saab_prev[i].Energy[j] < self.TH1: # this is a leaf node
                        continue
                    else: # this is an intermediate node
                        if train==True:
                            ct += 1
                        
                    self.split = True
                    X_tmp = X[ct].reshape(S)
                    
                    if train == True:
                        # fit
                        saab = self.SaabFit(X_tmp, layer=layer, bias=bias_prev[i])
                        # children node's energy should be multiplied by the parent's energy
                        saab.Energy *= saab_prev[i].Energy[j]
                        # remove the discarded nodes
                        saab = self.discard_nodes(saab)
                        # store
                        saab_cur.append(saab)
                        bias_cur.append(saab.Bias_current)
                        eng_cur.append(saab.Energy) 
                        # get output features
                        out_tmp = self.SaabTransform(X_tmp, saab=saab, layer=layer, train=True)
                    else:
                        out_tmp = self.SaabTransform(X_tmp, saab=saab_cur[pidx], layer=layer)
                        pidx += 1 # helping index
                        
                    output.append(out_tmp)
                    
                    # Clean the Cache
                    X_tmp = None
                    gc.collect()
                    out_tmp = None
                    gc.collect()
                    
            output = np.concatenate(output, axis=-1)
                    
        else: # saab, not channel-wise
            if train == True:
                # fit
                saab = self.SaabFit(X, layer=layer, bias=bias_prev[0])
                # remove the discarded nodes
                saab = self.discard_nodes(saab)
                # store
                saab_cur.append(saab)
                bias_cur.append(saab.Bias_current)
                eng_cur.append(saab.Energy)
                # get output features
                output = self.SaabTransform(X, saab=saab, layer=layer, train=True)
            else:
                output = self.SaabTransform(X, saab=saab_cur[0], layer=layer)

        if train == True:   
            if self.split == True or self.SaabArgs[0]['cw'] == False:
                self.par['Layer'+str(layer)] = saab_cur
                self.bias['Layer'+str(layer)] = bias_cur
                self.Energy['Layer'+str(layer)] = eng_cur
        
        return output
    
    def fit(self, X):
        '''train and learn cwsaab/saab kernels'''
        X = self.cwSaab_1_layer(X, train=True)
        print('=' * 45 + '>c/w Saab Train Hop 1')
        for i in range(1, self.depth):
            X = self.cwSaab_n_layer(X, train = True, layer = i)
            if (self.split == False and self.SaabArgs[i]['cw'] == True):
                self.depth = i
                print("       <WARNING> Cannot futher split, actual depth: %s"%str(i))
                break
            print('=' * 45 + f'>c/w Saab Train Hop {i+1}')
            self.split = False
        self.trained = True

    def transform(self, X):
        '''
        Get feature for all the Hops

        Parameters
        ----------
        X: Input image (N, H, W, C), C=1 for grayscale, C=3 for color image
        
        Returns
        -------
        output: a list of transformed feature maps
        '''
        assert (self.trained == True), "Must call fit first!"
        output = []
        X = self.cwSaab_1_layer(X, train = False)
        output.append(X)
        for i in range(1, self.depth):
            X = self.cwSaab_n_layer(X, train=False, layer=i)
            output.append(X)
            
        return output
    
    def transform_singleHop(self, X, layer=0):
        '''
        Get feature for a single Hop

        Parameters
        ----------
        X: previous Hops output (N, H1, W1, C1)
        layer: Hop index (start with 0)
        
        Returns
        -------
        output: transformed feature maps (N, H2, W2, C2)
        '''
        assert (self.trained == True), "Must call fit first!"
        if layer==0:
            output = self.cwSaab_1_layer(X, train = False)
        else:
            output = self.cwSaab_n_layer(X, train=False, layer=layer)
            
        return output
    
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn import datasets
    from skimage.util import view_as_windows
    
    # example callback function for collecting patches and its inverse
    def Shrink(X, shrinkArg):
        win = shrinkArg['win']
        stride = shrinkArg['stride']
        ch = X.shape[-1]
        X = view_as_windows(X, (1,win,win,ch), (1,stride,stride,ch))
        return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

    # read data
    print(" > This is a test example: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), 8, 8, 1))
    print(" input feature shape: %s"%str(X.shape))

    # set args
    SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'cw': False},
                {'num_AC_kernels':-1, 'needBias':True, 'cw':True}] 
    shrinkArgs = [{'func':Shrink, 'win':2, 'stride': 2}, 
                {'func': Shrink, 'win':2, 'stride': 2}]

    print(" -----> depth=2")
    cwsaab = cwSaab(depth=2, TH1=0.001,TH2=0.0005, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs)
    cwsaab.fit(X)
    output1 = cwsaab.transform(X)
    output2 = cwsaab.transform_singleHop(X)

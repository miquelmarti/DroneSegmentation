# Based on Evan Shelhamer's score.py from fcn.berkeleyvision.org

from __future__ import division
import numpy as np
import logging
logger = logging.getLogger()


def transplant(new_net, net):
    for p in net.params:
        if p not in new_net.params:
            logger.info('dropping %s', p)
            continue
        for i in range(len(net.params[p])):
            if i > (len(new_net.params[p]) - 1):
                logger.info('dropping %s %s', p, i)
                break
            if net.params[p][i].data.shape != new_net.params[p][i].data.shape:
                logger.info('coercing %s %s from %s to %s', p, i, net.params[p][i].data.shape, new_net.params[p][i].data.shape)
            else:
                logger.info('copying %s %s', p, i)
            new_net.params[p][i].data.flat = net.params[p][i].data.flat

            
def expand_score(new_net, new_layer, net, layer):
    old_cl = net.params[layer][0].num
    new_net.params[new_layer][0].data[:old_cl][...] = net.params[layer][0].data
    new_net.params[new_layer][1].data[0,0,0,:old_cl][...] = net.params[layer][1].data

    
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

    
def interp(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k and k != 1:
            logger.info('input + output channels need to be the same or |output| == 1')
            raise
        if h != w:
            logger.info('filters need to be square')
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

        
def fcnInterp(net):
    interp_layers = [k for k in net.params.keys() if 'up' in k]
    interp(net, interp_layers)

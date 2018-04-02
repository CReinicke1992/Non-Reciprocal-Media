#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 10:56:47 2018

@author: christianreini
"""

from Wavefield_NRM_p_w import Wavefield_NRM_p_w
import numpy as np
import sys

class Layered_NRM_p_w(Wavefield_NRM_p_w):
    
    def __init__(self,nt,dt,nr=1,dx=1,verbose=0,avec=np.zeros(1),bvec=np.zeros(1),g1vec=np.zeros(1),g3vec=np.zeros(1),p1=None,ReciprocalMedium=False):
        """
        defines a 1D (non-)reciprocal medium, and initialises a wavefield associated with a single ray-parameter in the ray-parameter frequency domain.
        
        The class Layered_NRM_p_w defines a 1.5D (non-)reciprocal medium and a scalar wavefield. We consider a single horizontal ray-parameter p1 and all frequencies that are sampled by the given number of time samples nt and the time sample interval dt.
        
        **Data format**:
         - Wavefields are saved in an array of dimensions (nt,nr).
         - Wavefields are in the p-`\\omega` domain.
         - The zero frequency component is placed at the first index position.
         - If the wavefield is transformed to the time domain, the zero time component is placed at the center of the time dimension.
        
        :param nt: Number of time samples
        :param dt: Time sample interval in seconds
        :param nr: Number of space samples
        :param dx: Space sample interval
        :param verbose: Set verbose=1 to receive feedback in the command line.
        :param avec: Medium parameter`\\alpha` (real-valued), must have the shape (n,) for n layers
        :param bvec: Medium parameter`\\beta` (real-valued), must have the shape (n,) for n layers
        :param g1vec: Medium parameter`\\gamma_1`  (real-valued?), must have the shape (n,) for n layers
        :param g3vec: Medium parameter`\\gamma_3`  (real-valued?), must have the shape (n,) for n layers
        :param p1: Horizontal ray-parameter in seconds per metre
        :param ReciprocalMedium: For non-reciprocal media set ReciprocalMedium=False, for reciprocal media set ReciprocalMedium=True 
        :type nt: int
        :type dt: int, float
        :type nr: int
        :type dx: int, float
        :type verbose: int
        :type avec: numpy.ndarray (real-valued)
        :type bvec: numpy.ndarray (real-valued)
        :type g1vec: numpy.ndarray (real-valued?)
        :type g3vec: numpy.ndarray (real-valued?)
        :type p1: int, float
        :type ReciprocalMedium: bool
        
        :Example:

        >>> from Layered_NRM_p_w import Layered_NRM_p_w as LM
        >>> F=LM(nt=1024,dt=0.005,avec=np.array([1,2,3]),bvec=np.array([0.4,3.14,2]),p1=2e-4,ReciprocalMedium=False)
        """
        
        # Inherit __init__ from Wavefield_NRM_p_w
        Wavefield_NRM_p_w.__init__(self,nt,dt,nr,dx)
        
        # Check if medium parameters are passed as arrays
        if not ( isinstance(avec,np.ndarray) and isinstance(bvec,np.ndarray) and isinstance(g1vec,np.ndarray) and isinstance(g3vec,np.ndarray) ):
            sys.exit('avec, bvec, g1vec and g3vec have to be of the type numpy.ndarray.')
            
        # Set gamma_1 and gamma_3 by default equal to zero
        if g1vec == np.zeros(1):
            g1vec = np.zeros_like(avec)
        if g3vec == np.zeros(1):
            g3vec = np.zeros_like(avec)
            
        # Force the medium parameters to have identical shape
        if avec.shape!=bvec.shape or avec.shape!=g1vec.shape or avec.shape!=g3vec.shape:
            sys.exit('avec, bvec, g1vec and g3vec have to be of identical shape.')
        
        # Force the medium parameters to be 1-dimensional, i.e. e.g. avec.shape=(n,)
        if avec.ndim!=1:
            sys.exit('avec.ndim, bvec.ndim, g1vec.ndim and g3vec.ndim must be one.')
            
        # Check if medium parameters correspond to a lossless (non-)reciprocal medium
        if ReciprocalMedium == False:
            if avec.imag.any()!=0 or bvec.imag.any()!=0 or g1vec.imag.any()!=0 or g3vec.imag.any()!=0:
                sys.exit('In lossless non-reciprocal media the imaginary value of avec, bvec, g1vec and g3vec has to be zero.')
        elif ReciprocalMedium == True:
            if avec.imag.any()!=0 or bvec.imag.any()!=0 or g1vec.real.any()!=0 or g3vec.real.any()!=0:
                sys.exit('In lossless reciprocal media the imaginary value of avec and bvec has to be zero, the real value of g1vec and g3vec has to be zero.')
            
        # Set medium parameters    
        self.avec = avec
        self.bvec = bvec
        self.g1vec = g1vec
        self.g3vec = g3vec
        self.p1 = p1
        self.ReciprocalMedium = ReciprocalMedium
            
        
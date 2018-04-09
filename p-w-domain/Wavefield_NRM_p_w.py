#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routines for modelling wavefields in 1D non-reciprocal media.

.. module:: Wavefield_NRM_p_w

:Authors:
    Christian Reinicke (c.reinicke@tudelft.nl), Kees Wapenaar (), and Evert Slob ()
    
:Copyright:
    Christian Reinicke (c.reinicke@tudelft.nl), Kees Wapenaar (), and Evert Slob ()
"""
        
import sys
import numpy as np
#import matplotlib.pylab as plt

class Wavefield_NRM_p_w:
    """is a class to define a scalar wavefield in the ray-parameter frequency domain.
        
    The class Wavefield_NRM_p_w defines the parameters of a scalar wavefield in a 1.5D (non-)reciprocal medium. We consider a single ray-parameter 'p1' and all frequencies that are sampled by the given number of time samples 'nt' and the time sample interval 'dt'.
        
    Parameters
    ----------

    nt : int
        Number of time samples.
    
    dt : int, float
        Time sample interval in seconds.
        
    nr : int, optional
        Number of space samples.
    
    dx : int, float, optional
        Space sample interval.
        
    verbose : bool, optional
        Set 'verbose=True' to receive feedback in the command line.
        
    Returns
    -------
    
    class
        A class to define a wavefield in a 1.5D non-reciprocal medium in the ray-parameter frequency domain.
    
    **Data format**:
     - Wavefields are saved in an array of dimensions (nt,nr).
     - Wavefields are in the p- :math:`\omega` domain.
     - The zero frequency component is placed at the first index position.
     - If the wavefield is transformed to the time domain, the zero time component is placed at the center of the time dimension.
     
    **Example:** Initialise a wavefield class
    
    >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
    >>> F=WF(nt=1024,dt=0.005)
    
    """
    
    def __init__(self,nt,dt,nr=1,dx=1,verbose=False):
        
        if type(nr) is not int:
            sys.exit('Wavefield_NRM_p_w: nr has to be an integer.')
        if type(nt) is not int:
            sys.exit('Wavefield_NRM_p_w: nt has to be an integer.')
        self.nt = nt
        self.dt = dt
        self.nr = nr
        self.dx = dx
        self.verbose = verbose
        self.nf = int(self.nt/2) + 1 # Index of Nyquist frequency + 1
        self.nk = int(self.nr/2) + 1 # Index of Nyquist space sample + 1
        self.author = "Christian Reinicke"
        
    # Frequency sampling    
    def Dw(self):
        """returns frequency sampling interval :math:`\Delta` :math:`\omega` in radians.
        
        The frequency sampling interval is defined by the time sampling interval :math:`\Delta t` and the number of time samples nt.
        
        Returns
        -------
        float
            Frequency sample interval in radians :math:`\Delta \omega`
        
        
        .. math:: 
            \Delta \omega = \frac{2 \pi}{\Delta t nt}
        
        
        **Example**

        >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
        >>> F=WF(nt=1024,dt=0.005)
        >>> F.Dw()
        1.227184630308513
        
        """
        return 2*np.pi/(self.dt*self.nt)
      
    # Frequency vector    
    def Wvec(self):
        """returns a vector of all frequency samples :math:`\omega` in radians.
        
        Returns
        -------
        dict
            Dictionary that contains the frequency vector with zero frequency placed 
                - at the center '**wvec**' (nt,1), 
                - at the first index position '**wvecfft**' (nt,1).
            
        
        **Example**

        >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
        >>> F=WF(nt=1024,dt=0.005)
        >>> F.Wvec()['wvec']
        array([[-628.31853072], ..., [ 627.09134609]])
    
        """
        dw = 2*np.pi/(self.dt*self.nt)
        wvec = np.zeros((self.nt,1))
        wvec[:,0] = dw*np.arange(-self.nt/2,self.nt/2)
        wvecfft = np.fft.ifftshift(wvec)
        
        out = {'wvec':wvec,'wvecfft':wvecfft}
        return out
    
    # Space vector
    def Xvec(self):
        """returns a vector of all spatial samples x in metres.
        
        Returns
        -------
        dict
            Dictionary that contains the offset vector with zero offset placed 
                - at the center '**xvec**' (nr,1), 
                - at the first index position '**xvecfft**' (nr,1).
                
        
        **Example**

        >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx=10)
        >>> F.Xvec()['xvec']
        array([[-2560.], ...,[ 2550.]])
        """
        xvec = np.zeros((self.nr,1))
        xvec[:,0] = self.dx*np.arange(-self.nr/2,self.nr/2)
        xvecfft = np.fft.ifftshift(xvec)
        
        out = {'xvec':xvec,'xvecfft':xvecfft}
        return out
    
    # Time vector
    def Tvec(self):
        """returns a vector of all time samples t in seconds.
        
        Returns
        -------
        dict
            Dictionary that contains the time vector with zero time placed 
                - at the center '**tvec**' (nt,1), 
                - at the first index position '**tvecfft**' (nt,1).
                
        
        **Example**

        >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx=10)
        >>> F.Tvec()['tvec']
        array([[-2.56 ], ..., [ 2.555]])
        """
        tvec = np.zeros((self.nt,1))
        tvec[:,0] = self.dt*np.arange(-self.nt/2,self.nt/2)
        tvecfft = np.fft.ifftshift(tvec)
        
        out = {'tvec':tvec,'tvecfft':tvecfft}
        return out
    
    # Make a 2D meshgrid in t-x-domain
    def T_X_grid(self):
        """returns two time-space meshgrids. 
        
         Returns
        -------
        dict
            Dictionary that contains a meshgrid
                - with the time vector tvec (the zero time sample is placed in the center) along the 1st dimension, and nr copies of it along the 2nd dimension '**Tgrid**' (nt,nr),
                - with the offset vector xvec (the zero offset sample is placed in the center) along the 2nd dimension, and nt copies of it along the 1st dimension '**Xgrid**' (nt,nr),
                - with the time vector tvecfft (the zero time sample is placed at the first index position) along the 1st dimension, and nr copies of it along the 2nd dimension '**Tgridfft**' (nt,nr),
                - with the offset vector xvecfft (the zero offset sample is placed at the first index position) along the 2nd dimension, and nt copies of it along the 1st dimension '**Xgridfft**' (nt,nr).
    
        
        **Example**

        >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx=10)
        >>> F.T_X_grid()['Tgridfft'][:,0]
        array([ 0.   ,  0.005,  0.01 , ..., -0.015, -0.01 , -0.005])
        """
        tvec = self.Tvec()['tvec']
        tvecfft = self.Tvec()['tvecfft']
        xvec = self.Xvec()['xvec']
        xvecfft = self.Xvec()['xvecfft']
        Xgrid,Tgrid = np.meshgrid(xvec,tvec)
        Xgridfft,Tgridfft = np.meshgrid(xvecfft,tvecfft)
        
        out = {'Tgrid':Tgrid,'Xgrid':Xgrid,'Tgridfft':Tgridfft,'Xgridfft':Xgridfft}
        return out
    
    # Make a 2D meshgrid in w-x-domain
    def W_X_grid(self):
        """returns two frequency-space meshgrids. 
        
        Returns
        -------
        dict
            Dictionary that contains a meshgrid
                - with the frequency vector wvec (the zero frequency sample is placed in the center) along the 1st dimension, and nr copies of it along the 2nd dimension '**Wgrid**' (nt,nr),
                - with the offset vector xvec (the zero offset sample is placed in the center) along the 2nd dimension, and nt copies of it along the 1st dimension '**Xgrid**' (nt,nr),
                - with the frequency vector wvecfft (the zero frequency sample is placed at the first index position) along the 1st dimension, and nr copies of it along the 2nd dimension '**Wgridfft**' (nt,nr),
                - with the offset vector xvecfft (the zero offset sample is placed at the first index position) along the 2nd dimension, and nt copies of it along the 1st dimension '**Xgridfft**' (nt,nr).
                
        
        :Example:

        >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx=10)
        >>> F.W_X_grid()['Wgridfft'][:,0]
        array([ 0.        ,  1.22718463,  2.45436926, ..., -3.68155389, -2.45436926, -1.22718463])
        """
        wvec = self.Wvec()['wvec']
        wvecfft = self.Wvec()['wvecfft']
        xvec = self.Xvec()['xvec']
        xvecfft = self.Xvec()['xvecfft']
        Xgrid,Wgrid = np.meshgrid(xvec,wvec)
        Xgridfft,Wgridfft = np.meshgrid(xvecfft,wvecfft)
        
        out = {'Wgrid':Wgrid,'Xgrid':Xgrid,'Wgridfft':Wgridfft,'Xgridfft':Xgridfft}
        return out
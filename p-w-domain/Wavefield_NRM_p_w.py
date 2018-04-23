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
        
    The class Wavefield_NRM_p_w defines the parameters of a scalar wavefield in a 1D (non-)reciprocal medium. We consider a single ray-parameter 'p1' and all frequencies that are sampled by the given number of time samples 'nt' and the time sample interval 'dt'.
        
    Parameters
    ----------

    nt : int
        Number of time samples.
    
    dt : int, float
        Time sample interval in seconds.
        
    nr : int, optional
        Number of space samples.
    
    dx1 : int, float, optional
        Space sample interval.
        
    verbose : bool, optional
        Set 'verbose=True' to receive feedback in the command line.
        
    Returns
    -------
    
    class
        A class to define a wavefield in a 1D non-reciprocal medium in the ray-parameter frequency domain. The following instances are defined:
            - **nt**: Number of time samples.
            - **dt**: Time sample interval in seconds.
            - **nr**: Number of space samples.
            - **dx1**: Number of space samples.
            - **verbose**: If one sets 'verbose=True' feedback will be output in the command line.
            - **nf**: Number of positive time samples :math:`=0.5 nt + 1`.
            - **nk**: Number of positive space samples :math:`=0.5 nr + 1`.
            
    
    Notes
    -----
    We format the data as described below.
        - Wavefields are saved in an array of dimensions (nf,nr) in the frequency domain and (nt,nr) in the time domain.
        - Wavefields are in the p- :math:`\omega` domain.
        - The zero frequency component is placed at the first index position.
        - If the wavefield is transformed to the time domain, the zero time component is placed at the first index position, followed by nt/2-1 positive time samples and nt/2 negative time samples.
     
    Examples
    -------- 
    
    >>> # Initialise a wavefield class
    >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
    >>> F=WF(nt=1024,dt=0.005)
    
    """
    
    def __init__(self,nt,dt,nr=1,dx1=1,verbose=False):
        
        if type(nr) is not int:
            sys.exit('Wavefield_NRM_p_w: nr has to be an integer.')
        if type(nt) is not int:
            sys.exit('Wavefield_NRM_p_w: nt has to be an integer.')
        if (type(dt) is not int) and (type(dt) is not float):
            sys.exit('Wavefield_NRM_p_w: dt has to be an integer or a float.')
        if (type(dx1) is not int) and (type(dx1) is not float):
            sys.exit('Wavefield_NRM_p_w: dx1 has to be an integer or a float.')
        if nt<=0 or nr<=0 or dt<=0 or dx1<=0:
            sys.exit('Wavefield_NRM_p_w: nt, nr, dt and dx1 must be greater than zero.')
        # Check if verbose is a bool
        if not isinstance(verbose,bool):
            sys.exit('Wavefield_NRM_p_w: \'verbose\' must be of the type bool.')
            
        self.nt = nt
        self.dt = dt
        self.nr = nr
        self.dx1 = dx1
        self.verbose = verbose
        self.nf = int(self.nt/2) + 1 # Index of Nyquist frequency + 1
        self.nk = int(self.nr/2) + 1 # Index of Nyquist space sample + 1
        self.author = "Christian Reinicke"
        
    # Frequency sampling    
    def Dw(self):
        """returns frequency sampling interval :math:`\Delta` :math:`\omega` in radians.
        
        The frequency sampling interval is defined by the time sampling interval :math:`\Delta t` and the number of time samples 'nt'. 
        
        Returns
        -------
        
        float
            Frequency sample interval in radians :math:`\Delta \omega` :math:`= \\frac{2 \pi}{\Delta t \; nt}`
        
        
        Examples
        --------
        
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
        numpy.ndarray
            Frequency vector, zero frequency is placed at the first index position. The vector has the shape (nf,1).
        
        Examples
        --------

        >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
        >>> F=WF(nt=1024,dt=0.005)
        >>> F.Wvec()
        array([[0.], ..., [628.31853072]])
    
        """
        dw = 2*np.pi/(self.dt*self.nt)
        wvec = np.zeros((self.nf,1))
        wvec[:,0] = dw*np.arange(0,self.nf)
 
        return wvec
    
    # Space vector
    def Xvec(self):
        """returns a vector of all spatial samples :math:`x` in metres.
        
        Returns
        -------
        dict
            Dictionary that contains the offset vector,
                - **xvec**: zero offset placed at the center.
                - **xvecfft**: zero offset placed at the first index position.
            Both vectors have the shape (nr,1).
                

        Examples
        --------

        >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx1=10)
        >>> F.Xvec()['xvec']
        array([[-2560.], ...,[ 2550.]])
        
        """
        xvec = np.zeros((self.nr,1))
        xvec[:,0] = self.dx1*np.arange(-self.nr/2,self.nr/2)
        xvecfft = np.fft.ifftshift(xvec)
        
        out = {'xvec':xvec,'xvecfft':xvecfft}
        return out
    
    # Time vector
    def Tvec(self):
        """returns a vector of all time samples :math:`t` in seconds.
        
        Returns
        -------
        dict
            Dictionary that contains the time vector, 
                - **tvec**: zero time placed at the center. 
                - **tvecfft**: zero time placed at the first index position.
            Both vectors have the shape (nt,1).
                
        
        Examples
        --------

        >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx1=10)
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
                - **Tgrid**: with the time vector *tvec* (the zero time sample is placed in the center) along the 1st dimension, and nr copies of it along the 2nd dimension.
                - **Xgrid**: with the offset vector *xvec* (the zero offset sample is placed in the center) along the 2nd dimension, and nt copies of it along the 1st dimension.
                - **Tgridfft**: with the time vector *tvecfft* (the zero time sample is placed at the first index position) along the 1st dimension, and nr copies of it along the 2nd dimension.
                - **Xgridfft**: with the offset vector *xvecfft* (the zero offset sample is placed at the first index position) along the 2nd dimension, and nt copies of it along the 1st dimension.
            All output arrays have the shape (nt,nr).
    
        
        Examples
        --------

        >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx1=10)
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
                - **Wgrid**: with the frequency vector *wvec* (the zero frequency sample is placed in the center) along the 1st dimension, and nr copies of it along the 2nd dimension.
                - **Xgrid**: with the offset vector *xvec* (the zero offset sample is placed in the center) along the 2nd dimension, and nt copies of it along the 1st dimension.
                - **Xgridfft**: with the offset vector *xvecfft* (the zero offset sample is placed at the first index position) along the 2nd dimension, and nt copies of it along the 1st dimension.
            All output arrays have the shape (nf,nr).
                
        
        Examples
        --------

        >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx1=10)
        >>> F.W_X_grid()['Wgrid'][:,0]
        array([ 0.        ,  1.22718463,  2.45436926, ..., 628.3185307179587])
        
        """
        wvec = self.Wvec()
        xvec = self.Xvec()['xvec']
        xvecfft = self.Xvec()['xvecfft']
        Xgrid,Wgrid = np.meshgrid(xvec,wvec)
        Xgridfft,_ = np.meshgrid(xvecfft,wvec)
        
        out = {'Wgrid':Wgrid,'Xgrid':Xgrid,'Xgridfft':Xgridfft}
        return out
    
    # Transform wavefield from w-p-domain to t-p-domain
    def PW2PT(self,array_pw,NumPy_fft_Sign_Convention=False):
        """applies an inverse Fourier transform from the :math:`p_1`-:math:`\omega` domain to  the :math:`p_1`-:math:`t` domain. 
        
        We assume that the time domain signal is real-valued (:math:`p_1`-:math:`t` domain). Therefore, we use the NumPy function :py:class:`numpy.fft.irfft`.
        
        Parameters
        ----------
    
        array_pw : numpy.ndarray
            Array in the :math:`p_1`-:math:`\omega` domain, shape (nf,nr).
        
        NumPy_fft_Sign_Convention : bool, optional
            Set 'NumPy_fft_Sign_Convention=True' if Numpy's sign convention is used for the inverse Fourier transform (positive sign in the exponential of the inverse Fourier transform). 
            
        Returns
        -------
        
        numpy.ndarray
            Real-valued array in the :math:`p_1`-:math:`t` domain, shape (nt,nr).
            
        Notes
        -----
        
        In the sub-class **Layered_NRM_p_w** we define the wavefield propagators with a positive sign, :math:`\\tilde{w}^{\pm} = \mathrm{exp}(+j \omega p_3' \Delta x_3)`. Thus, we implicitly assume that the inverse Fourier transfrom is defined with a negative sign in the exponential function, which is why we set by default 'NumPy_fft_Sign_Convention=False'.
    
        """
        
        if not isinstance(array_pw,np.ndarray):
            sys.exit('PW2PT: The input variable \'array_pw\' must be of the type numpy.ndarray.')
            
        if array_pw.shape != (self.nf,self.nr):
            sys.exit('PW2PT: The input variable \'array_pw\' must have the shape (nf,nr).')
        
        # If ifft sign convention is opposite to NumPy's convention
        # We apply a complex conjugation first, such that NumPy's ifft can
        # be applied as inverse Fourier transform
        if NumPy_fft_Sign_Convention is False:
            array_pw = array_pw.conj()
            
        array_pt = np.fft.irfft(array_pw,n=None,axis=0,norm=None)
        
        return array_pt
    
    # Transform wavefield from p-t-domain to p-w-domain
    def PT2PW(self,array_pt,NumPy_fft_Sign_Convention=False):
        """applies a forward Fourier transform from the :math:`p_1`-:math:`t` domain to the :math:`p_1`-:math:`\omega` domain. 
        
        We assume that the time domain signal is real-valued (:math:`p_1`-:math:`t` domain). Therefore, we use the NumPy function :py:class:`numpy.fft.rfft`.
        
        Parameters
        ----------
    
        array_pt : numpy.ndarray
            Real-valued array in the :math:`p_1`-:math:`t` domain, shape (nt,nr).
        
        NumPy_fft_Sign_Convention : bool, optional
            Set 'NumPy_fft_Sign_Convention=True' if Numpy's sign convention is used for the Fourier transform (negative sign in the exponential of the forward Fourier transform). 
            
        Returns
        -------
        
        numpy.ndarray
            Array in the :math:`p_1`-:math:`\omega` domain, shape (nf,nr).
            
        Notes
        -----
        
        In the sub-class **Layered_NRM_p_w** we define the wavefield propagators with a positive sign, :math:`\\tilde{w}^{\pm} = \mathrm{exp}(+j \omega p_3' \Delta x_3)`. Thus, we implicitly assume that the forward Fourier transfrom is defined with a positive sign in the exponential function, which is why we set by default 'NumPy_fft_Sign_Convention=False'.
    
        """
        
        if not isinstance(array_pt,np.ndarray):
            sys.exit('PT2PW: The input variable \'array_pt\' must be of the type numpy.ndarray.')
            
        if array_pt.shape != (self.nt,self.nr):
            sys.exit('PT2PW: The input variable \'array_pt\' must have the shape (nt,nr).')
            
        array_pw = np.fft.rfft(array_pt,n=None,axis=0,norm=None)
        
        # If fft sign convention is opposite to NumPy's convention
        # We apply a complex conjugation, such that NumPy's fft can
        # be applied as forward Fourier transform
        if NumPy_fft_Sign_Convention is False:
            array_pw = array_pw.conj()
        
        return array_pt
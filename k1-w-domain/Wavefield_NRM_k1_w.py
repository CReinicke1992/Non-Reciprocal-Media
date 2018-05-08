#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routines for modelling wavefields in 1.5D non-reciprocal media.

.. module:: Wavefield_NRM_k1_w

:Authors:
    Christian Reinicke (c.reinicke@tudelft.nl), Kees Wapenaar (), and Evert Slob ()
    
:Copyright:
    Christian Reinicke (c.reinicke@tudelft.nl), Kees Wapenaar (), and Evert Slob ()
"""
        
import sys
import numpy as np
#import matplotlib.pylab as plt

class Wavefield_NRM_k1_w:
    """is a class to define a scalar wavefield in the horizontal-wavenumber frequency domain (:math:`k_1-\omega`).
        
    The class Wavefield_NRM_k1_w defines the parameters of a scalar wavefield in a 1.5D (non-)reciprocal medium. We consider all horizontal-wavenumbers and all frequencies, that are sampled by the given number of samples and by the given sample intervals, in space ('nr', 'dx1') as well as in time ('nt', 'dt').
        
    Parameters
    ----------

    nt : int
        Number of time samples.
    
    dt : int, float
        Time sample interval in seconds.
        
    nr : int
        Number of space samples.
    
    dx1 : int, float
        Space sample interval in metres.
        
    verbose : bool, optional
        Set 'verbose=True' to receive feedback in the command line.
        
    eps : int, float, optional
        A real-valued scalar can be assigned to 'eps' to reduce the wrap-around effect of wavefields in the time domain. If the inverse Fourier transform is defined as,
            :math:`f(t)  = \int F(\omega) \; \mathrm{e}^{\mathrm{j} \omega t} \mathrm{d}\omega`,
        which is ensured if the function **K1W2X1T** is used, 'eps'(:math:`=\epsilon`) should be positive to the suppress wrap-around effect from positive to negative time,
            :math:`f(t) \mathrm{e}^{- \epsilon t} = \int F(\omega + \mathrm{j} \epsilon) \; \mathrm{e}^{\mathrm{j} (\omega + \mathrm{j} \epsilon) t} \mathrm{d}\omega`.
        Recommended value eps = :math:`\\frac{3 nf}{dt}`.
            
        
    Returns
    -------
    
    class
        A class to define a wavefield in a 1.5D non-reciprocal medium in the horizontal-wavenumber frequency domain. The following instances are defined:
            - **nt**: Number of time samples.
            - **dt**: Time sample interval in seconds.
            - **nr**: Number of space samples.
            - **dx1**: Number of space samples.
            - **verbose**: If one sets 'verbose=True' feedback will be output in the command line.
            - **eps**: Scalar constant to reduce temporal wrap-around effect.
            - **nf**: Number of positive time samples :math:`=0.5 nt + 1`.
            - **nk**: Number of positive space samples :math:`=0.5 nr + 1`.
            
    Notes
    -----
    We format the data as described below.
        - Wavefields are saved in an array of dimensions (nf,nr) in the frequency domain and (nt,nr) in the time domain.
        - Wavefields are in the :math:`k_1`-:math:`\omega` domain.
        - The zero frequency component is placed at the first index position of the first dimension.
        - The zero horizontal-wavenumber component is placed at the first index position of the second dimension.
        - If the wavefield is transformed to the space-time domain: 
            - The zero time component is placed at the first index position of the first dimension, followed by nt/2-1 positive time samples and nt/2 negative time samples. 
            - The zero offset component is placed at the first index position of the second dimension, followed by nr/2-1 positive offset samples and nr/2 negative offset samples.
            
     
    Examples
    -------- 
    
    >>> # Initialise a wavefield class
    >>> from Wavefield_NRM_k1_w import Wavefield_NRM_k1_w as WF
    >>> F=WF(nt=1024,dt=0.005,nr=512,dx1=5)
    
    """
    
    def __init__(self,nt,dt,nr,dx1,verbose=False,eps=None):
        
        if (type(nr) is not int):
            sys.exit('Wavefield_NRM_k1_w: nr has to be an integer.')
        if type(nt) is not int:
            sys.exit('Wavefield_NRM_k1_w: nt has to be an integer.')
        if (type(dt) is not int) and (type(dt) is not float):
            sys.exit('Wavefield_NRM_k1_w: dt has to be an integer or a float.')
        if (type(dx1) is not int) and (type(dx1) is not float):
            sys.exit('Wavefield_NRM_k1_w: dx1 has to be an integer or a float.')
        if nt<=0 or nr<=0 or dt<=0 or dx1<=0:
            sys.exit('Wavefield_NRM_k1_w: nt, nr, dt and dx1 must be greater than zero.')
        # Check if verbose is a bool
        if not isinstance(verbose,bool):
            sys.exit('Wavefield_NRM_k1_w: \'verbose\' must be of the type bool.')
        if eps is not None: 
            if (type(eps) is not int) and (type(eps) is not float):
                sys.exit('Wavefield_NRM_k1_w: \'eps\' must be of the type int or float.')
            
        self.nt = nt
        self.dt = dt
        self.nr = nr
        self.dx1 = dx1
        self.verbose = verbose
        self.eps = eps
        self.nf = int(self.nt/2) + 1 # Index of Nyquist frequency + 1
        self.nk = int(self.nr/2) + 1 # Index of Nyquist space sample + 1
        self.author = "Christian Reinicke"
        
    # Horizontal-wavenumber sampling    
    def Dk1(self):
        """returns horizontal-wavenumber sampling interval :math:`\Delta k_1` in :math:`\mathrm{m}^{-1}`.
        
        The horizontal-wavenumber sampling interval is defined by the space sampling interval :math:`\Delta x_1` ('dx1') and the number of space samples 'nr'. 
        
        Returns
        -------
        
        float
            Horizontal-wavenumber sample interval in :math:`\mathrm{m}^{-1}` :math:`\Delta k_1` :math:`= \\frac{2 \pi}{\Delta x_1 \; nr}`.
        
        
        Examples
        --------
        
        >>> from Wavefield_NRM_k1_w import Wavefield_NRM_k1_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx1=5)
        >>> F.Dk1()
        0.002454369260617026
        
        """
        return 2*np.pi/(self.dx1*self.nr)
    
    # Horizontal-wavenumber vector    
    def K1vec(self):
        """returns a vector of all horizontal-wavenumber samples :math:`k_1` in :math:`\mathrm{m}^{-1}`.
        
        Returns
        -------
        dict
            Dictionary that contains the horizontal-wavenumber vector:
                - **k1vec**: nr/2 negative horizontal-wavenumbers are placed in the first part, followed by zero horizontal-wavenumber (index position nr/2), and nr/2-1 positive horizontal-wavenumbers.
                - **k1vecfft**: zero horizontal-wavenumber is placed at the first index position, followed by nr/2-1 positive horizontal-wavenumbers, and nr/2 negative horizontal-wavenumbers.
           The vector has the shape (nr,1). The Nyquist component is negative.
        
        Examples
        --------

        >>> from Wavefield_NRM_k1_w import Wavefield_NRM_k1_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx1=5)
        >>> F.K1vec()['k1vecfft'][F.nk-1,0]
        -0.6283185307179586
    
        """
        dk1 = self.Dk1()
        k1vec = np.zeros((self.nr,1))
        k1vec[:,0] = dk1*np.arange(-self.nr/2,self.nr/2)
        k1vecfft = np.fft.ifftshift(k1vec)
        
        out = {'k1vec':k1vec,'k1vecfft':k1vecfft}
        return out
    
    # Frequency sampling    
    def Dw(self):
        """returns frequency sampling interval :math:`\Delta` :math:`\omega` in radians.
        
        The frequency sampling interval is defined by the time sampling interval :math:`\Delta t` ('dt') and the number of time samples 'nt'. 
        
        Returns
        -------
        
        float
            Frequency sample interval in radians :math:`\Delta \omega` :math:`= \\frac{2 \pi}{\Delta t \; nt}`
        
        
        Examples
        --------
        
        >>> from Wavefield_NRM_k1_w import Wavefield_NRM_k1_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx1=5)
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
            Frequency vector, zero frequency is placed at the first index position. The vector has the shape (nf,1). If 'self.eps' is defined, an imaginary constant :math:`\epsilon` is added to the frequency vector. 
        
        Examples
        --------

        >>> from Wavefield_NRM_k1_w import Wavefield_NRM_k1_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx1=5)
        >>> F.Wvec()
        array([[0.], ..., [628.31853072]])
    
        """
        dw = 2*np.pi/(self.dt*self.nt)
        wvec = np.zeros((self.nf,1))
        wvec[:,0] = dw*np.arange(0,self.nf)
        
        # Add imaginary constant to the frequency
        if self.eps is not None:
            wvec = wvec + 1j*self.eps
 
        return wvec
    
    # Make a 2D meshgrid in w-k1-domain
    def W_K1_grid(self):
        """returns frequency horizontal-wavenumber meshgrids. 
        
        Returns
        -------
        dict
            Dictionary that contains a meshgrid
                - **Wgrid**: with the frequency vector *wvec* (the zero frequency sample is placed at the first index position) along the 1st dimension, and nr copies of it along the 2nd dimension.
                - **Xgrid**: with the offset vector *xvec* (the zero offset sample is placed in the center) along the 2nd dimension, and nt copies of it along the 1st dimension.
                - **Xgridfft**: with the offset vector *xvecfft* (the zero offset sample is placed at the first index position) along the 2nd dimension, and nt copies of it along the 1st dimension.
            All output arrays have the shape (nf,nr).
                
        
        Examples
        --------

        >>> from Wavefield_NRM_k1_w import Wavefield_NRM_k1_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx1=5)
        >>> F.W_K1_grid()['Wgrid'][:,0]
        array([ 0.        ,  1.22718463,  2.45436926, ..., 628.3185307179587])
        
        """
        wvec = self.Wvec()
        k1vec = self.K1vec()['k1vec']
        k1vecfft = self.K1vec()['k1vecfft']
        K1grid,Wgrid = np.meshgrid(k1vec,wvec)
        K1gridfft,_ = np.meshgrid(k1vecfft,wvec)
        
        out = {'Wgrid':Wgrid,'K1grid':K1grid,'K1gridfft':K1gridfft}
        return out
    
    # Space vector
    def Xvec(self):
        """returns a vector of all spatial samples :math:`x_1` in metres.
        
        Returns
        -------
        dict
            Dictionary that contains the offset vector,
                - **xvec**: zero offset placed at the center.
                - **xvecfft**: zero offset placed at the first index position.
            Both vectors have the shape (nr,1).
                

        Examples
        --------

        >>> from Wavefield_NRM_k1_w import Wavefield_NRM_k1_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx1=5)
        >>> F.Xvec()['xvec']
        array([[-1280.], ...,[ 1275.]])
        
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

        >>> from Wavefield_NRM_k1_w import Wavefield_NRM_k1_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx1=5)
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
        """returns time-space meshgrids. 
        
        Returns
        -------
        dict
            Dictionary that contains the meshgrid
                - **Tgrid**: with the time vector *tvec* (the zero time sample is placed in the center) along the 1st dimension, and nr copies of it along the 2nd dimension.
                - **Xgrid**: with the offset vector *xvec* (the zero offset sample is placed in the center) along the 2nd dimension, and nt copies of it along the 1st dimension.
                - **Tgridfft**: with the time vector *tvecfft* (the zero time sample is placed at the first index position) along the 1st dimension, and nr copies of it along the 2nd dimension.
                - **Xgridfft**: with the offset vector *xvecfft* (the zero offset sample is placed at the first index position) along the 2nd dimension, and nt copies of it along the 1st dimension.
            All output arrays have the shape (nt,nr).
    
        
        Examples
        --------

        >>> from Wavefield_NRM_k1_w import Wavefield_NRM_k1_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx1=5)
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
                - **Wgrid**: with the frequency vector *wvec* (the zero frequency sample is placed at the first index position) along the 1st dimension, and nr copies of it along the 2nd dimension.
                - **Xgrid**: with the offset vector *xvec* (the zero offset sample is placed in the center) along the 2nd dimension, and nt copies of it along the 1st dimension.
                - **Xgridfft**: with the offset vector *xvecfft* (the zero offset sample is placed at the first index position) along the 2nd dimension, and nt copies of it along the 1st dimension.
            All output arrays have the shape (nf,nr).
                
        
        Examples
        --------

        >>> from Wavefield_NRM_k1_w import Wavefield_NRM_k1_w as WF
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
    
    # Transform wavefield from k1-w-domain to x1-t-domain
    def K1W2X1T(self,array_k1w,NumPy_fft_Sign_Convention=False,K1W_fft_Same_Sign=False):
        """applies an inverse Fourier transform from the :math:`k_1`-:math:`\omega` domain to  the :math:`x_1`-:math:`t` domain. 
        
        We assume that the space-time domain signal is real-valued (:math:`x_1`-:math:`t` domain). Therefore, we use the NumPy function :py:class:`numpy.fft.irfftn`.
        
        Parameters
        ----------
    
        array_k1w : numpy.ndarray
            Array in the :math:`k_1`-:math:`\omega` domain, shape (nf,nr).
        
        NumPy_fft_Sign_Convention : bool, optional
            Set 'NumPy_fft_Sign_Convention=True' if the temporal inverse Fourier transform is defined with same sign convention as Numpy's :py:class:`numpy.fft.ifft`: :math:`\int \cdot \; \mathrm{e}^{+\mathrm{j} \omega t} \mathrm{d}\omega`. 
            
        K1W_fft_Same_Sign : bool, optional
            - Set 'K1W_fft_Same_Sign=True' to apply the temporal and the spatial inverse Fourier tansform with the same signs in the exponential: :math:`\int\int \cdot \; \mathrm{e}^{\pm(\mathrm{j} \omega t + \mathrm{j} k_1 x_1)} \mathrm{d}k_1 \; \mathrm{d}\omega`.
            - Set 'K1W_fft_Same_Sign=False' to apply the temporal and the spatial inverse Fourier tansform with opposite signs in the exponential: :math:`\int\int \cdot \; \mathrm{e}^{\pm(\mathrm{j} \omega t - \mathrm{j} k_1 x_1)} \mathrm{d}k_1 \; \mathrm{d}\omega`.
            
        Returns
        -------
        
        numpy.ndarray
            Real-valued array in the :math:`x_1`-:math:`t` domain, shape (nt,nr).
            
        Notes
        -----
        
            - In the sub-class **Layered_NRM_k1_w**, we define the wavefield propagators with a positive sign, :math:`\\tilde{w}^{\pm} = \mathrm{exp}(+j k_3'(\omega) \Delta x_3)`. Thus, we implicitly assume that the temporal inverse Fourier transfrom is defined with a negative sign in the exponential function, which is why we set by default 'NumPy_fft_Sign_Convention=False'.
        
            - By default, we set 'K1W_fft_Same_Sign=False', i.e. the default definition of the inverse Fourier transform is,
            \t\t:math:`f(x_1,t)= \\frac{1}{(2\pi)^2} \int \int F(k_1,\omega) \; \mathrm{e}^{-\mathrm{j}\omega t + \mathrm{j} k_1 x_1} \; \mathrm{d}\omega \; \mathrm{d}k_1`.
                                                                                                                    
        """
        
        if not isinstance(array_k1w,np.ndarray):
            sys.exit('K1W2X1T: The input variable \'array_k1w\' must be of the type numpy.ndarray.')
            
        if array_k1w.shape != (self.nf,self.nr):
            sys.exit('K1W2X1T: The input variable \'array_k1w\' must have the shape (nf,nr).')
        
        # If the sign convention for the inverse temporal Fourier transform  is 
        # opposite to NumPy's ifft convention, we complex-conjugate the 
        # wavefield before applying an ifft.
        if NumPy_fft_Sign_Convention is False:
            array_k1w = array_k1w.conj()
            
        # If the temporal and spatial Fourier transforms are defined with 
        # opposite signs, we reverse the horizontal-wavenumber dimension.
        if K1W_fft_Same_Sign is False:
            array_k1w[:,1:] = array_k1w[:,-1:0:-1]
            
        # Numpy's irfftn applies a standard ifftn along along specified axes,
        # and an irfft along the last specified axis.
        # Here, we want to apply the irfft along time/frequency (1st axis).
        # Thus, we apply a transpose before and after applying irfftn.
        array_x1t = np.fft.irfftn(array_k1w.T,s=None,axes=(0,1),norm=None).T
        
        return array_x1t
    
    # Transform wavefield from x1-t-domain to k1-w-domain
    def X1T2K1W(self,array_x1t,NumPy_fft_Sign_Convention=False,K1W_fft_Same_Sign=False):
        """applies a forward Fourier transform from the :math:`x_1`-:math:`t` domain to the :math:`k1_1`-:math:`\omega` domain. 
        
        We assume that the space-time domain signal is real-valued (:math:`x_1`-:math:`t` domain). Therefore, we use the NumPy function :py:class:`numpy.fft.rfftn`.
        
        Parameters
        ----------
    
        array_pt : numpy.ndarray
            Real-valued array in the :math:`x_1`-:math:`t` domain, shape (nt,nr).
            
        NumPy_fft_Sign_Convention : bool, optional
            Set 'NumPy_fft_Sign_Convention=True' if the temporal forward Fourier transform is defined with same sign convention as Numpy's :py:class:`numpy.fft.fft`: :math:`\int \cdot \; \mathrm{e}^{-\mathrm{j} \omega t} \mathrm{d}t`. 
            
        K1W_fft_Same_Sign : bool, optional
            - Set 'K1W_fft_Same_Sign=True' to apply the temporal and the spatial forward Fourier tansform with the same signs in the exponential: :math:`\int\int \cdot \; \mathrm{e}^{\pm(\mathrm{j} \omega t + \mathrm{j} k_1 x_1)} \mathrm{d}x_1 \; \mathrm{d}t`.
            - Set 'K1W_fft_Same_Sign=False' to apply the temporal and the spatial forward Fourier tansform with opposite signs in the exponential: :math:`\int\int \cdot \; \mathrm{e}^{\pm(\mathrm{j} \omega t - \mathrm{j} k_1 x_1)} \mathrm{d}x_1 \; \mathrm{d}t`.
            
        Returns
        -------
        
        numpy.ndarray
            Array in the :math:`k_1`-:math:`\omega` domain, shape (nf,nr).
            
        Notes
        -----
        
            - In the sub-class **Layered_NRM_k1_w** we define the wavefield propagators with a positive sign, :math:`\\tilde{w}^{\pm} = \mathrm{exp}(+j k_3'(\omega) \Delta x_3)`. Thus, we implicitly assume that the temporal forward Fourier transfrom is defined with a positive sign in the exponential, which is why we set by default 'NumPy_fft_Sign_Convention=False'.
        
            - By default, we set ‘K1W_fft_Same_Sign=False’, i.e. the default definition of the forward Fourier transform is,
            \t\t:math:`F(k_1,\omega)=\int \int f(x_1,\omega) \mathrm{e}^{+\mathrm{j}\omega t - \mathrm{j} k_1 x_1} \; \mathrm{d}t \; \mathrm{d}x_1`.
    
        """
        
        if not isinstance(array_x1t,np.ndarray):
            sys.exit('X1T2K1W: The input variable \'array_x1t\' must be of the type numpy.ndarray.')
            
        if array_x1t.shape != (self.nt,self.nr):
            sys.exit('X1T2K1W: The input variable \'array_x1t\' must have the shape (nt,nr).')
            
        # Numpy's rfftn applies a standard fftn along along specified axes,
        # and an rfft along the last specified axis.
        # Here, we want to apply the rfft along time/frequency (1st axis).
        # Thus, we apply a transpose before and after applying rfftn.
        array_k1w = np.fft.rfftn(array_x1t.T,s=None,axes=(0,1),norm=None).T
        
        # If the sign convention for the forward temporal Fourier transform  is 
        # opposite to NumPy's fft convention, we introduced an error above.
        # To correct the error, we complex-conjugate the result.
        if NumPy_fft_Sign_Convention is False:
            array_k1w = array_k1w.conj()
            
        # If the temporal and spatial Fourier transforms are defined with 
        # opposite signs, we reverse the horizontal-wavenumber dimension.
        if K1W_fft_Same_Sign is False:
            array_k1w[:,1:] = array_k1w[:,-1:0:-1]
        
        return array_k1w
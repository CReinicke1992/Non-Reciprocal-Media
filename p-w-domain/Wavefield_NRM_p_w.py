#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 14:43:03 2018

@author: christianreini
"""
# Example for a docstring
"""returns (arg1 / arg2) + arg3

        This is a longer explanation, which may include math with latex syntax
        :math:`\\alpha`.
        Then, you need to provide optional subsection in this order (just to be
        consistent and have a uniform documentation. Nothing prevent you to
        switch the order):

          - parameters using ``:param <name>: <description>``
          - type of the parameters ``:type <name>: <description>``
          - returns using ``:returns: <description>``
          - examples (doctest)
          - seealso using ``.. seealso:: text``
          - notes using ``.. note:: text``
          - warning using ``.. warning:: text``
          - todo ``.. todo:: text``

        **Advantages**:
         - Uses sphinx markups, which will certainly be improved in future
           version
         - Nice HTML output with the See Also, Note, Warnings directives


        **Drawbacks**:
         - Just looking at the docstring, the parameter, type and  return
           sections do not appear nicely

        :param arg1: the first value
        :param arg2: the first value
        :param arg3: the first value
        :type arg1: int, float,...
        :type arg2: int, float,...
        :type arg3: int, float,...
        :returns: arg1/arg2 +arg3
        :rtype: int, float

        :Example:

        >>> import template
        >>> a = template.MainClass1()
        >>> a.function1(1,1,1)
        2

        .. note:: can be useful to emphasize
            important feature
        .. seealso:: :class:`MainClass2`
        .. warning:: arg2 must be non-zero.
        .. todo:: check that arg2 is non zero.
        """
        
import sys
import numpy as np
import matplotlib.pylab as plt

class Wavefield_NRM_p_w:
    
    def __init__(self,nt,dt,nr=1,dx=1,verbose=0):
        """
        initialises a wavefield in a 1D non-reciprocal medium in the ray-parameter frequency domain.
        
        The class Wavefield_NRM_p_w defines the parameters of a scalar wavefield in a 1.5D non-reciprocal medium. We consider a single ray-parameter p and all frequencies that are captured by the given number of time samples nt and the time sample interval dt.
        
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
        :type nt: int
        :type dt: int, float
        :type nr: int
        :type dx: int, float
        :type verbose: int
        
        :Example:

        >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
        >>> F=WF(nt=1024,dt=0.005)
        """
        if type(nr) is not int:
            sys.exit('nr has to be an integer.')
        if type(nt) is not int:
            sys.exit('nt has to be an integer.')
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
        """
        returns frequency sampling interval `\\Delta \\omega`in radians.
        
        The frequency sampling interval is defined by the time sampling interval `\\Delta t` and the number of time samples nt.
        
        :return: `\\frac{2 \\pi}{\\Delta t nt}`
        :rtype: float
        
        :Example:

        >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
        >>> F=WF(nt=1024,dt=0.005)
        >>> F.Dw()
        1.227184630308513
        """
        return 2*np.pi/(self.dt*self.nt)
      
    # Frequency vector    
    def Wvec(self):
        """
        returns a vector of all frequency samples `\\omega` in radians.
        
        :return wvec: Frequency vector with zero frequency placed at the center of the vector
        :return wvecfft: Frequency vector with zero frequency placed the first index position of the vector
        :rtype: dict
        
        :Example:

        >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
        >>> F=WF(nt=1024,dt=0.005)
        >>> F.Wvec()['wvec']
        array([[-628.31853072],
       ...,
       [ 627.09134609]])
        """
        dw = 2*np.pi/(self.dt*self.nt)
        wvec = np.zeros((self.nt,1))
        wvec[:,0] = dw*np.arange(-self.nt/2,self.nt/2)
        wvecfft = np.fft.ifftshift(wvec)
        
        out = {'wvec':wvec,'wvecfft':wvecfft}
        return out
    
    # Space vector
    def Xvec(self):
        """
        returns a vector of all spatial samples x in metres.
        
        :return xvec: Offset vector with zero offset placed at the center of the vector
        :return xvecfft: Offset vector with zero offset placed the first index position of the vector
        :rtype: dict
        
        :Example:

        >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx=10)
        >>> F.Xvec()['xvec']
        array([[-2560.],
       ...,
       [ 2550.]])
        """
        xvec = np.zeros((self.nr,1))
        xvec[:,0] = self.dx*np.arange(-self.nr/2,self.nr/2)
        xvecfft = np.fft.ifftshift(xvec)
        
        out = {'xvec':xvec,'xvecfft':xvecfft}
        return out
    
    # Time vector
    def Tvec(self):
        """
        returns a vector of all time samples t in seconds.
        
        :return tvec: Time vector with time zero placed at the center of the vector
        :return tvecfft: Time vector with time zero placed the first index position of the vector
        :rtype: dict
        
        :Example:

        >>> from Wavefield_NRM_p_w import Wavefield_NRM_p_w as WF
        >>> F=WF(nt=1024,dt=0.005,nr=512,dx=10)
        >>> F.Tvec()['tvec']
        array([[-2.56 ],
       ...,
       [ 2.555]])
        """
        tvec = np.zeros((self.nt,1))
        tvec[:,0] = self.dt*np.arange(-self.nt/2,self.nt/2)
        tvecfft = np.fft.ifftshift(tvec)
        
        out = {'tvec':tvec,'tvecfft':tvecfft}
        return out
    
    # Make a 2D meshgrid in t-x-domain
    def T_X_grid(self):
        """
        returns two time-space meshgrids. 
        
        :return Tgrid: Meshgrid with the time vector tvec - hence the zero time sample is placed in the center - along the 1st dimension and nr copies of it along the 2nd dimension
        :return Xgrid: Meshgrid with the space vector xvec - hence the zero offset sample is placed in the center - along the 2nd dimension and nt copies of it along the 1st dimension
        :return Tgridfft: Meshgrid with the time vector tvecfft - hence the zero time sample is placed at the first index position - along the 1st dimension and nr copies of it along the 2nd dimension
        :return Xgridfft: Meshgrid with the space vector xvecfft - hence the zero offset sample is placed at the first index position - along the 2nd dimension and nt copies of it along the 1st dimension
        :rtype: dict
        
        :Example:

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
        """
        returns two frequency-space meshgrids. 
        
        :return Wgrid: Meshgrid with the frequency vector wvec - hence the zero frequency sample is placed in the center - along the 1st dimension and nr copies of it along the 2nd dimension
        :return Xgrid: Meshgrid with the space vector xvec - hence the zero offset sample is placed in the center - along the 2nd dimension and nt copies of it along the 1st dimension
        :return Wgridfft: Meshgrid with the time vector wvecfft - hence the zero frequency sample is placed at the first index position - along the 1st dimension and nr copies of it along the 2nd dimension
        :return Xgridfft: Meshgrid with the space vector xvecfft - hence the zero offset sample is placed at the first index position - along the 2nd dimension and nt copies of it along the 1st dimension
        :rtype: dict
        
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
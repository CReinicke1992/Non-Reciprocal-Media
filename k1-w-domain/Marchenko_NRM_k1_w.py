#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routines for applying the Marchenko method in 1.5D non-reciprocal media.

.. module:: Marchenko_NRM_k1_w 2\.0

:Authors:
    Christian Reinicke (c.reinicke@tudelft.nl), Kees Wapenaar (), and Evert Slob ()
    
:Copyright:
    Christian Reinicke (c.reinicke@tudelft.nl), Kees Wapenaar (), and Evert Slob ()
"""

from Layered_NRM_k1_w import Layered_NRM_k1_w
import numpy as np
import sys

class Marchenko_NRM_k1_w(Layered_NRM_k1_w):
    """is a class to apply the Marchenko method to scalar wavefields in 1.5D 
    (non-)reciprocal media.
    
    The class Marchenko_NRM_k1_w operates on scalar wavefields in 1.5D (non-)
    reciprocal media, defined by the class Layered_NRM_k1_w. We consider all 
    horizontal-wavenumbers and all frequencies, that are sampled by the given 
    number of samples and by the given sample intervals, in space ('nr', 'dx1') 
    as well as in time ('nt', 'dt'). Note that the separation of the focusing 
    and Green's functions requires an inverse Fourier transformation to the 
    space-time domain. By applying an inverse Fourier transformation, followed 
    by a truncation in the space-time domain and a forward Fourier 
    transformation we introduce small artefacts due to finite sampling. Thus, 
    the focusing functions cannot be retrieved within double-precision.
    
    
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
        A real-valued scalar can be assigned to 'eps' to reduce the wrap-around 
        effect of wavefields in the time domain. If the inverse Fourier 
        transform is defined as,
            :math:`f(t)  = \int F(\omega) \; \mathrm{e}^{\mathrm{j} \omega t} 
            \mathrm{d}\omega`,
        which is ensured if the function **K1W2X1T** is used, 'eps' 
        (:math:`=\epsilon`) should be positive to the suppress wrap-around 
        effect from positive to negative time,
            :math:`f(t) \mathrm{e}^{- \epsilon t} = 
            \int F(\omega + \mathrm{j} \epsilon) \; 
            \mathrm{e}^{\mathrm{j} (\omega + \mathrm{j} \epsilon) t} 
            \mathrm{d}\omega`.
        Recommended value eps = :math:`\\frac{3}{n_f dt}`.
        
    x3vec : numpy.ndarray
        Vertical spatial vector :math:`x_3`, for n layers 'x3vec' must have the 
        shape (n,). We define the :math:`x_3`-axis as downward-pointing. 
        Implicitly, the first value on the :math:`x_3`-axis is zero (not stored 
        in 'x3vec').
    
    avec : numpy.ndarray
        Medium parameter :math:`\\alpha` (real-valued), for n layers 'avec' 
        must have the shape (n,).
        
    b11vec : numpy.ndarray
        Medium parameter :math:`\\beta_{11}` (real-valued), for n layers 
        'b11vec' must have the shape (n,).
        
    b13vec : numpy.ndarray, optional
        Medium parameter :math:`\\beta_{13}` (real-valued), for n layers 
        'b13vec' must have the shape (n,).
        
    b33vec : numpy.ndarray
        Medium parameter :math:`\\beta_{33}` (real-valued), for n layers 
        'b33vec' must have the shape (n,).
    
    g1vec : numpy.ndarray, optional
        Medium parameter :math:`\gamma_1` (real-valued for non-reciprocal 
        media), for n layers 'g1vec' must have the shape (n,).
        
    g3vec : numpy.ndarray, optional
        Medium parameter :math:`\gamma_3` (real-valued for non-reciprocal 
        media), for n layers 'g3vec' must have the shape (n,).
        
    ReciprocalMedium : bool, optional
        For non-reciprocal media set 'ReciprocalMedium=False', for reciprocal 
        media set 'ReciprocalMedium=True'.
        
    AdjointMedium : bool, optional
        Set 'AdjointMedium=True' to compute scattering coefficients and 
        propagators in an adjoint medium :math:`^{(a)}`. We have defined the 
        eigenvectors in the adjoint medium only for flux-normalisation.
        
    x3F : int, float, optional
        Focusing depth.
    
        
    Returns
    -------
    
    class
        A class to apply the Marchenko method to a scalar wavefield in  a 1.5D 
        non-reciprocal medium. The following instances are 
        defined:
            
            - **x3vec**: :math:`x_3`.
            - **avec**: :math:`\\alpha`.
            - **b11vec**: :math:`\\beta_{11}`.
            - **b13vec**: :math:`\\beta_{13}`.
            - **b33vec**: :math:`\\beta_{33}`.
            - **g1vec**: :math:`\gamma_1`.
            - **g3vec**: :math:`\gamma_3`.
            - **ReciprocalMedium**: True for reciprocal media, False for non-reciprocal media.
            - **AdjointMedium**: If True, propagation and scattering are defined in a medium and in its adjoint.
            - **k3**: Vertical-wavenumber for positive 'k1'.
            - **k3n**: Vertical-wavenumber for negative 'k1'.
            - **x3F**: Focusing depth.
            
            
    References
    ----------
    
    Kees document as soon as it is published.
    
    
    
    """
    
    def __init__(self,nt,dt,nr,dx1,verbose=False,eps=None,x3vec=np.zeros(1),
                 avec=np.zeros(1),b11vec=np.zeros(1),b13vec=np.zeros(1),
                 b33vec=np.zeros(1),g1vec=np.zeros(1),g3vec=np.zeros(1),
                 ReciprocalMedium=False,AdjointMedium=False,x3F=None):
        
        # Inherit __init__ from Layered_NRM_k1_w
        Layered_NRM_k1_w.__init__(self,nt,dt,nr,dx1,verbose,eps,x3vec,avec,
                                  b11vec,b13vec,b33vec,g1vec,g3vec,
                                  ReciprocalMedium,AdjointMedium)
        
        # Set Marchenko parameters
        self.x3F = x3F
        self.P   = None
        
    def TruncateMedium(self,x3F,UpdateSelf=False):
        """truncates the medium below the focusing depth :math:`x_{3,f}` 
        ('x3F'). 
        
        
        Parameters
        ----------
        
        x3F : int, float
            Focusing depth.
            
        UpdateSelf : bool, optional
            If 'UpdateSelf=True' the focusing depth level is not only inserted 
            in the truncated medium but also in the 'self' parameters of the
            actual medium.
            
            
        Returns
        -------
        
        dict
            Dictionary that contains the parameters of the truncated medium
            
                - **x3vec**:  Depth vector.
                - **avec**:   :math:`\\alpha` vector.
                - **b11vec**: :math:`\\beta_{11}` vector.
                - **b13vec**: :math:`\\beta_{11}` vector.
                - **b33vec**: :math:`\\beta_{11}` vector.
                - **g1vec**:  :math:`\gamma_1` vector.
                - **g3vec**:  :math:`\gamma_3` vector.
                - **K3**:     :math:`k_3(k_1)` vector.
                - **K3n**:    :math:`k_3(-k_1)` vector.
                - **LP**:     Eigenvalues :math:`\lambda^+(k_1)`.
                - **LPn**:    Eigenvalues :math:`\lambda^+(-k_1)`.
                - **LM**:     Eigenvalues :math:`\lambda^-(k_1)`.
                - **LMn**:    Eigenvalues :math:`\lambda^-(-k_1)`.
                
            All medium parameter vectors are stored in arrays of shape (n,).
            The vertical wavenumbers and the eigenvalues are stored in arrays 
            of shape (nf,nr,n).
            
            If the eigenvalues were not computed previously the parameters 'LP', 
            'LM', 'LPn' and 'LMn' will be of the type None.    
    
        
        References
        ----------
        
        Kees document as soon as it is published.
        
            
        Examples
        --------
        
        >>> from Marchenko_NRM_k1_w import Marchenko_NRM_k1_w as Marchenko
        >>> import numpy as np
        
        >>> # Initialise wavefield
        >>> F = LM(nt=1024,dt=5e-3,nr=4096,dx1=12.5,
        >>>        x3vec=np.array([1.1,2.2,3.7])*1e3,eps=3/(513*5e-3),
        >>>        avec=np.array([1,2,3])*1e-3,
        >>>        b11vec=np.array([1.4,3.14,2])*1e-4,
        >>>        b13vec=np.array([0.4,2.4,1.2])*1e-4,
        >>>        b33vec=np.array([1.4,3.14,2])*1e-4,
        >>>        g1vec=np.array([0.8,2,1.3])*1e-4,
        >>>        g3vec=np.array([1.8,0.7,2.3])*1e-4,
        >>>        ReciprocalMedium=False,AdjointMedium=True)
        
        >>> F.x3vec
        array([1100., 2200., 3700.])
        
        >>> F.avec
        array([0.001, 0.002, 0.003])
        
        >>> # Truncate medium for focusing at x3F = 1500
        >>> Trunc = F.TruncateMedium(x3F=1500)
        
        >>> Trunc['x3vec']
        array([1100., 1500., 2200.])
        
        >>> Trunc['avec']
        array([0.001, 0.002, 0.002])
        
        """
        
        # Check if x3F is a positive scalar.
        if not ( isinstance(x3F,int) or isinstance(x3F,float) ) or x3F<0:
            sys.exit('TruncateMedium: The input variable \'x3F\' must be a '
                     +'scalar greater than, or equal to, zero.')
            
        # Check if UpdateSelf is a bool    
        if not isinstance(UpdateSelf,bool):
            sys.exit('TruncateMedium: The input variable \'UpdateSelf\' must '
                     +'be a bool.')
        
        # Insert focusing depth level
        # If docusing depth level is at the bottom of the medium, insert an
        # additional depth level below to avoid indexing errors
        if x3F >= self.x3vec[-1]:    
            TruncatedMedium = self.Insert_layer(x3=np.array([x3F,x3F+1]),
                                                UpdateSelf=UpdateSelf)
        else:
            TruncatedMedium = self.Insert_layer(x3=x3F,UpdateSelf=UpdateSelf)
        
        # Find truncation index
        X3  = TruncatedMedium['x3vec']
        f = X3.tolist().index(x3F)
        
        # Truncate medium
        X3  = X3[:f+2]
        A   = TruncatedMedium['avec'][:f+2]
        B11 = TruncatedMedium['b11vec'][:f+2]
        B13 = TruncatedMedium['b13vec'][:f+2]
        B33 = TruncatedMedium['b33vec'][:f+2]
        G1  = TruncatedMedium['g1vec'][:f+2]
        G3  = TruncatedMedium['g3vec'][:f+2]
        K3  = TruncatedMedium['K3'][:,:,:f+2]
        K3n = TruncatedMedium['K3n'][:,:,:f+2]
        
        LP  = TruncatedMedium['LP'] 
        LM  = TruncatedMedium['LM'] 
        LPn = TruncatedMedium['LPn'] 
        LMn = TruncatedMedium['LMn'] 
        
        if LP is not None:
            LP = LP[:,:,:f+2]
            LM = LM[:,:,:f+2]
        if LPn is not None:
            LPn = LPn[:,:,:f+2]
            LMn = LMn[:,:,:f+2]
        
        # Update medium parameters
        if UpdateSelf is True:
            self.x3F = x3F
        
        out = {'x3vec':X3,'avec':A,'b11vec':B11,'b13vec':B13,'b33vec':B33,
               'g1vec':G1,'g3vec':G3,'K3':K3,'K3n':K3n,
               'LP':LP,'LPn':LPn,'LM':LM,'LMn':LMn}
        
        return out
        
    def Projector_x1_t(self,x3F,f0=None,delta=0,RelativeTaperLength=2**(-6),
                       UpdateSelf=False,normalisation='flux'):
        """computes a projector that separates the focusing and Green's 
        functions in the space-time domain.
        
        The projector mutes all arrivals before the direct transmission 
        :math:`T_d^+(x_{3,f},x_{3,0},t)` associated with sources at the 
        surface :math:`x_{3,0}` and a receiver at the focusing point 
        :math:`x_{3,f}`. The direct transmission itself is also muted.
        
        
        Parameters
        ----------
        
        x3F : int, float
            Focusing depth.
            
        f0 : int, float
            Central frequency of a Ricker wavelet in Hz to compute the width of
            the taper of the projector according to,
            :math:`\\frac{\sqrt{6}}{\pi f_0}`
            (trough-to-trough time).
            
        delta : int, float
            The width of the wavelet in seconds that the projector should take
            into account. If the input parameter 'f0' is defined the parameter 
            'delta' is ignored.
            
        RelativeTaperLength : int, float, optional
            The product of \'RelativeTaperLength\' and the number of temporal 
            samples \'nt\' determines the taper length. The default value is 
            \'RelativeTaperLength\':math:`=2^{-5}`. 
            
        UpdateSelf : bool, optional
            If 'UpdateSelf=True' the focusing depth level 'x3F' and the 
            projector 'P' are saved as 'self' parameters. In addition, the 
            focusing depth level is inserted in the model.
            
        normalisation : str, optional
            For pressure-normalisation set normalisation='pressure', for 
            flux-normalisation set normalisation='flux'. 
            
            
        Returns
        -------
        
        dict
            Dictionary that contains 
            
                - **P**:    The projector.
                - **td**:   The direct transmission.
                - **Pa**:   The projector (adjoint medium).
                - **tda**:  The direct transmission (adjoint medium).
                
            The outputs are in the space-time domain. They are stored in arrays 
            of shape (nt,nr).


        Examples
        --------
        
        >>> from Marchenko_NRM_k1_w import Marchenko_NRM_k1_w as Marchenko
        >>> import numpy as np
        
        >>> # Initialise wavefield
        >>> F = LM(nt=1024,dt=5e-3,nr=4096,dx1=12.5,
        >>>        x3vec=np.array([1.1,2.2,3.7])*1e3,eps=0,
        >>>        avec=np.array([1,2,3])*1e-3,
        >>>        b11vec=np.array([1.4,3.14,2])*1e-4,
        >>>        b13vec=np.array([0.4,2.4,1.2])*1e-4,
        >>>        b33vec=np.array([1.4,3.14,2])*1e-4,
        >>>        g1vec=np.array([0.8,2,1.3])*1e-4,
        >>>        g3vec=np.array([1.8,0.7,2.3])*1e-4,
        >>>        ReciprocalMedium=False,AdjointMedium=True)
        
        >>> # Compute projector for focusing at 1500m depth
        >>> x3F=1500
        >>> out=F.Projector_x1_t(x3F,f0=30,RelativeTaperLength=2**(-5),
        >>>                      UpdateSelf=False,normalisation='flux')
        >>> P = out['P']
        >>> # We plot the projector below.
        
        .. image:: ../pictures/cropped/Projector.png
           :width: 200px
           :height: 200px
        
        >>> # We test how efficiently the projector mutes the direct 
        >>> # transmission td
        >>> td = out['td']
        >>> np.linalg.norm(P*td)
        5.4907520410514596e-05
        >>> np.linalg.norm(P*td)/np.linalg.norm(td)
        0.0002078217409067099
        
        """
        # Check if x3F is a positive scalar.
        if not ( isinstance(x3F,int) or isinstance(x3F,float) ) or x3F<0:
            sys.exit('Projector_x1_t: The input variable \'x3F\' must be a '
                     +'scalar greater than, or equal to, zero.')
        
        # Check the RelativeTaperLength
        if not ( isinstance(RelativeTaperLength,int) 
              or isinstance(RelativeTaperLength,float) ):
            sys.exit('Projector_x1_t: \'RelativeTaperLength\' must be of the '
                     +'type int or float.')
            
        # Check that RelativeTaperLength is not smaller than zero
        if RelativeTaperLength < 0:
            sys.exit('Projector_x1_t: \'RelativeTaperLength\' must be greater '
                     +'than, or equal to zero.')
        
        # Check if UpdateSelf is a bool    
        if not isinstance(UpdateSelf,bool):
            sys.exit('Projector_x1_t: The input variable \'UpdateSelf\' must '
                     +'be a bool.')
        
        # Check if normalisation is set correctly
        if (normalisation is not 'flux') and (normalisation is not 'pressure'):
            sys.exit('Projector_x1_t: The input variable \'normalisation\' '
                     +'must be set, either to \'flux\', or to \'pressure\'.')
            
        # Determine shift of the projector onset
        if f0 is not None:
            # Check the peak frequency f0
            if not ( isinstance(f0,int) or isinstance(f0,float) ):
                sys.exit('Projector_x1_t: \'f0\' must be of the type int or '
                         +'float.')
            # Ensure f0 is not smaller than zero
            if f0 < 0:
                sys.exit('Projector_x1_t: \'f0\' must be greater than, or '
                         +'equal to zero.')     
            
            # Shift of the projector onset in sample-numbers
            s = int(np.sqrt(6)/(np.pi*f0*self.dt))
            
        else:
            # Check the time shift of the projector onset 'delta'
            if not ( isinstance(delta,int) or isinstance(delta,float) ):
                sys.exit('Projector_x1_t: \'delta\' must be of the type int or '
                         +'float.')
            # Ensure that 'delta' is not smaller than zero
            if delta < 0:
                sys.exit('Projector_x1_t: \'delta\' must be greater than, or '
                         +'equal to zero.') 
                
            # Shift of the projector onset in sample-numbers
            s = int(delta/self.dt)
        
        # Construct the taper of the projector
        taperlen = int(RelativeTaperLength*self.nt)
        if taperlen != 0:
            tap = np.cos(np.arange(0,taperlen)*0.5*np.pi/(taperlen-1))**2

        # Model the direct transmission
        T  = self.TruncateMedium(x3F=x3F,UpdateSelf=UpdateSelf)
        eps = 3/(self.nf*self.dt)
        Resp = self.RT_response_k1_w(x3vec=T['x3vec'],avec=T['avec'],
                                     b11vec=T['b11vec'],b13vec=T['b13vec'],
                                     b33vec=T['b33vec'],g1vec=T['g1vec'],
                                     g3vec=T['g3vec'],eps=eps,
                                     normalisation=normalisation,
                                     InternalMultiples=False)
        
        # Make a zero-phase wavelet for a more coherent projector
        if f0 is None:
            f0 = self.Dw()*self.nf/(2*np.pi*5)
        Wav = self.RickerWavelet_w(f0=f0,eps=eps)
        
        # Gain function to correct for complex-valued frequency
        gain = self.Gain_t(RelativeTaperLength=0,eps=eps)
        
        # Direct transmission in the space-time domain
        Td = Resp['TP']
        td = np.fft.fftshift(gain*self.K1W2X1T(Td*Wav),axes=0)
        
        # Initialise the projector
        P = np.ones((self.nt,self.nr))
        cut = int(self.nf/4)
        
        # Iterate over offsets and pick first arrivals until the last time 
        # sample is reached
        for x1 in range(self.nr):
            ind = np.argmax(np.abs(td[:,x1]))-s
            if ind < cut:
                break
            P[ind:,x1] = 0
            P[ind-taperlen:ind,x1] = tap
        
        nx = x1    
        
        # Iterate over negative offsets (x1>=0) and search for first arrival
        for x1 in np.arange(self.nr-1,nx,-1):
            ind = np.argmax(np.abs(td[:,x1])) - s
            if ind < cut:
                break
            P[ind:,x1] = 0
            P[ind-taperlen:ind,x1] = tap
        
        P = np.fft.ifftshift(P,axes=0)
        td = np.fft.ifftshift(td,axes=0)
        
        if self.AdjointMedium is True:
            Tda = Resp['TPa']
            tda = np.fft.fftshift(gain*self.K1W2X1T(Tda*Wav),axes=0)
        
            # Initialise the projector
            Pa = np.ones((self.nt,self.nr))
            
            # Iterate over offsets and pick first arrivals until the last time 
            # sample is reached
            for x1 in range(self.nr):
                ind = np.argmax(np.abs(tda[:,x1]))-s
                if ind < cut:
                    break
                Pa[ind:,x1] = 0
                Pa[ind-taperlen:ind,x1] = tap
            
            nx = x1    
            
            # Iterate over negative offsets (x1>=0) and search for first arrival
            for x1 in np.arange(self.nr-1,nx,-1):
                ind = np.argmax(np.abs(tda[:,x1])) - s
                if ind < cut:
                    break
                Pa[ind:,x1] = 0
                Pa[ind-taperlen:ind,x1] = tap
            
            Pa = np.fft.ifftshift(Pa,axes=0)
            tda = np.fft.ifftshift(tda,axes=0)
        
        if UpdateSelf is True:
            self.x3F = x3F
            self.P   = P
        
        out={'P':P,'td':td,'Pa':Pa,'tda':tda}
        return out
    
    def F_initial_k1_w(self,x3F,f0=None,FK_filter=False,
                       RelativeTaperLength=2**(-5),normalisation='flux',
                       UpdateSelf=False):
        """computes the initial downgoing focusing function 
        :math:`F_1^+(k_1,x_{3,0},x_{3,f},\omega)`. The temporal wrap-around 
        is already reduced to provide a good initial focusing function.
        
        
        Parameters
        ----------
        
        x3F : int, float
            Focusing depth.
            
        f0 : int, float, optional
            Central frequency of a Ricker wavelet in Hz. If not defined the 
            initial focusing function is not multiplied by a wavelet.
            
        FK_filter : bool, optional
            Set 'FK_filter=True' to apply a :math:`k_1-\omega` filter to the 
            intial focusing function. The :math:`k_1-\omega` filter is defined
            by the fastest propagation velocity of the truncated medium.
            
        RelativeTaperLength : int, float, optional
            The :math:`k_1-\omega` filter can be tapered to avoid sharp edges.
            To this end, define the \'RelativeTaperLength\' which is the 
            quotient of the number of time samples of the taper and the number 
            of temporal samples \'nt\'. The default value is 
            \'RelativeTaperLength\':math:`=2^{-5}.`
            
        normalisation : str, optional
            For pressure-normalisation set normalisation='pressure', for 
            flux-normalisation set normalisation='flux'. We use the function
            'FocusingFunction_k1_w' which until now only works for flux-
            normalisation.
            
        UpdateSelf : bool, optional
            If 'UpdateSelf=True' the focusing depth level 'x3F' is inserted in 
            the model and the focusing depth is saved as self-parameter.
            
            
        Returns
        -------
        
        dict
            Dictionary that contains 
            
                - **FP0**:    Downgoing focusing function.
                - **FP0a**    Downgoing focusing function in the adjoint medium.
                
            The outputs are in the wavenumber-frequency domain. They are stored 
            in arrays of shape (nf,nr).


        Examples
        --------
        
        >>> from Marchenko_NRM_k1_w import Marchenko_NRM_k1_w as Marchenko
        >>> import numpy as np
        
        >>> # Initialise wavefield
        >>> F = Marchenko(nt=1024,dt=5e-3,nr=4096,dx1=12.5,
        >>>              x3vec=np.array([1.1,2.2,3.7])*1e3,
        >>>              avec=np.array([1,2,3])*1e-3,
        >>>              b11vec=np.array([1.4,3.14,2])*1e-4,
        >>>              b13vec=np.array([0.4,2.4,1.2])*1e-4,
        >>>              b33vec=np.array([1.4,3.14,2])*1e-4,
        >>>              g1vec=np.array([0.8,2,1.3])*1e-4,
        >>>              g3vec=np.array([1.8,0.7,2.3])*1e-4,
        >>>              ReciprocalMedium=False,AdjointMedium=True)
        
        >>> # Compute initial focusing function for a focusing depth at 1500m
        >>> Focus=F.F_initial_k1_w(x3F=1500,f0=30,FK_filter=True,
        >>>                        RelativeTaperLength=0,
        >>>                        normalisation='flux')
        >>>                      UpdateSelf=False,normalisation='flux')
        >>> FP0 = Focus['FP0']
        >>> # We plot the initial focusing function below.
        
        .. image:: ../pictures/cropped/InitialF1plus_k1_w.png
           :width: 200px
           :height: 200px
        
        >>> # We transform the initial focusing function to the space-time 
        >>> # domain
        >>> fP0 = F.K1W2X1T(FP0)
        >>> # We plot the initial focusing function below.
           
        .. image:: ../pictures/cropped/InitialF1plus_x1_t.png
            :width: 200px
            :height: 200px
        
        """
        # Check if FK_filter is bool
        if not isinstance(FK_filter,bool):
            sys.exit('F_initial_k1_w: The variable F_filter must be a bool.')
            
        # Ensure to suppress wrap-around effects for the initial estimate
        eps = -3/(self.nf*self.dt)
        self.SubSelf = Layered_NRM_k1_w(self.nt,self.dt,self.nr,self.dx1,
                                        self.verbose,eps=eps,
                                        x3vec=self.x3vec,avec=self.avec,
                                        b11vec=self.b11vec,
                                        b13vec=self.b13vec,
                                        b33vec=self.b33vec,
                                        g1vec=self.g1vec,g3vec=self.g3vec,
                                        ReciprocalMedium=self.ReciprocalMedium,
                                        AdjointMedium=self.AdjointMedium)
        
        # Compute direct downgoing focusing function
        Focus = self.SubSelf.FocusingFunction_k1_w(x3F, normalisation='flux',
                                                   InternalMultiples=False,
                                                   UpdateSelf=True)
        FP  = Focus['FP']
        FPa = Focus['FPa']

        # If f0 is given make Ricker wavelet if f0
        if f0 is not None:
            Wav = self.SubSelf.RickerWavelet_w(f0=f0)
        else:
            Wav = np.ones_like(FP)
            
        # If FK_filter is true make FK filter based on highest velocity of
        # the truncated medium
        if FK_filter is True:    
            f = self.SubSelf.x3vec.tolist().index(x3F)
            Mask = self.SubSelf.FK1_mask_k1_w(RelativeTaperLength=
                                              RelativeTaperLength)
            ind = np.argmin(np.linalg.norm(Mask['FK'][:,:,:f+1],axis=(0,1)))
            FK  = Mask['FK_tap'][:,:,ind]
            FKn = Mask['FKn_tap'][:,:,ind]
            del Mask
        else:
            FK  = np.ones_like(FP)
            FKn = np.ones_like(FP)
        
        # Apply wavelet and FK filter to the focusing function
        FP  = Wav*FK*FP
        FPa = Wav*FKn*FPa
        
        # Compute correction for complex-valued frequency
        gain = self.SubSelf.Gain_t()
    
        # Apply correction
        FP = self.X1T2K1W(gain*self.K1W2X1T(FP))
        FPa = self.X1T2K1W(gain*self.K1W2X1T(FPa))
        
        # Update self
        if UpdateSelf is True:
            if x3F >= self.x3vec[-1]:
                Tmp_medium = self.Insert_layer(x3=np.array([x3F,x3F+1]),
                                               UpdateSelf=UpdateSelf)
            else:
                Tmp_medium = self.Insert_layer(x3=x3F,UpdateSelf=UpdateSelf)
        
            self.x3F = x3F
            
        out = {'FP0':FP,'FP0a':FPa}
        return out
    
    def MarchenkoSeries(self,x3F,K,R=None,FP0=None,P=None,f0=None,delta=0,
                        FK_filter=False,RelativeTaperLength=2**(-5),
                        normalisation='flux',UpdateSelf=False,AdjointF=False):
        """computes K iterations of the Marchenko series.
        
        
        Parameters
        ----------
        
        x3F : int, float
            Focusing depth.
            
        K : int, float
            Number of iterations of the Marchenko series.
            
        R : numpy.ndarray, optional
            Reflection response of the actual medium in the wavenumber-
            frequency domain (nf,nr).
        
        FP0 : numpy.ndarray, optional
            Initial focusing function in the wavenumber-frequency domain 
            (nf,nr).
            
        P : numpy.ndarray, optional
            Projector in the space-time domain (nt,nr).
            
        f0 : int, float, optional
            If 'f0' is defined and 'FP0=None' the initial focusing function is 
            multiplied by a Ricker wavelet with peak frequency f0. If 'f0' is 
            defined and 'P=None' the projector onset is shifted by the width of 
            a Ricker wavelet with peak frequency f0.
            
        delta : int, float, optional
            If 'P=None' and 'f0=None', the width of the wavelet in seconds that 
            the projector should take into account can be given by 'delta'.
            
        FK_filter : bool, optional
            If 'FP0=None' and 'FK_filter=True' an :math:`k_1-\omega` filter is 
            applied to the intial focusing function. The :math:`k_1-\omega` 
            filter is defined by the fastest propagation velocity of the 
            truncated medium.
            
        RelativeTaperLength : int, float, optional
            The product of the RelativeTaperLength and the number of time 
            samples 'nt' determines the lenght of the taper that is applied
            to the :math:`k_1-\omega` if 'FP0=None' and 'FK_filter=True', to 
            the projector onset if 'P=None', and the amplitude correction of 
            the initial focusing function if 'FP0=None'.
            
        normalisation : str, optional
            For pressure-normalisation set normalisation='pressure', for 
            flux-normalisation set normalisation='flux'. If 'FP0=None' the 
            initial focusing function is modelled by 'FocusingFunction_k1_w' 
            which until now only works for flux-normalisation.
            
        UpdateSelf : bool, optional
            If 'UpdateSelf=True' the focusing depth level 'x3F' is inserted in 
            the model and the focusing depth is saved as self-parameter.
            
        AdjointF : bool, optional
            For non-reciprocal media there are two sets of Marchenko equations:
            Set 'AdjointF=True' to solve the Marchenko equations associated
            with the focusing functions in the adjoint medium.
            Set 'AdjointF=False' to solve the Marchenko equations associated
            with the focusing functions in the actual medium.
            
            
        Returns
        -------
        
        dict
            Dictionary that contains 
            
                - **FP**:    Downgoing focusing function.
                - **FM**:    Downgoing focusing function in the adjoint medium.
                - **FP0**    Initial focusing function.
                _ **P**      Projector that mutes the Green's functions.
                - **R**      Reflection response.
                
                 
            The outputs are associated with the actual medium if 'AdjointF=
            False' or with the adjoint medium if 'AdjointF=True'. All outputs,
            except the projector **P**, are in the wavenumber-frequency domain. 
            They are stored in arrays of shape (nf,nr). The projector is stored
            in an array of shape (nt,nr).


        Examples
        --------
        
        >>> from Marchenko_NRM_k1_w import Marchenko_NRM_k1_w as Marchenko
        >>> import numpy as np
        
        >>> # Initialise wavefield
        >>> F = Marchenko(nt=4096,dt=5e-3,nr=2048,dx1=12.5,
        >>>              x3vec=np.array([1.1,2.2,3.7])*1e3,
        >>>              avec=np.array([2,4,6])*1e-3,
        >>>              b11vec=np.array([1.4,3.14,2])*1e-4,
        >>>              b13vec=np.array([0.4,2.4,1.2])*1e-4,
        >>>              b33vec=np.array([1.4,3.14,2])*1e-4,
        >>>              g1vec=np.array([0.8,2,1.3])*1e-4,
        >>>              g3vec=np.array([1.8,0.7,2.3])*1e-4,
        >>>              ReciprocalMedium=False,AdjointMedium=True)
        
        >>> # Compute initial focusing function for a focusing depth at 3000m
        >>> Focus=F.MarchenkoSeries(x3F=3000,K=5,R=None,FP0=None,P=None,f0=30,
        >>>                         delta=0,
        >>>                         FK_filter=True,RelativeTaperLength=2**(-6),
        >>>                         normalisation='flux',UpdateSelf=False,
        >>>                         AdjointF=False)
        >>> FP = Focus['FP']
        >>> fP = F.K1W2X1T(FP)
        
        >>> # Reference focusing function
        >>> Modelled = F.FocusingFunction_k1_w(x3F,normalisation='flux',
                                   InternalMultiples=True,Negative_eps=False,
                                   UpdateSelf=False)
        >>> FPmod = Modelled['FP']
        >>> Wav  = F.RickerWavelet_w(f0=30)
        >>> Mask = F.FK1_mask_k1_w(RelativeTaperLength=2**(-6))
        >>> ind  = np.argmin(np.linalg.norm(Mask['FK'][:,:,:3],axis=(0,1)))
        >>> FK   = Mask['FK_tap'][:,:,ind]
        >>> fPmod = F.K1W2X1T(FPmod*FK*Wav)
        
        >>> # We plot the retrieved focusing function and the difference below.
        
        .. image:: ../pictures/cropped/RetrievedF1plus_x1_t.png
           :width: 300px
           :height: 200px
        
        
        """
        # Check if K is an integer larger than or equal to zero
        if (not isinstance(K,int)) or K <0:
            sys.exit('MarchenkoSeries: \'K\' must be an integer greater than, '
                     +'or equal to, zero.')
        
        # Check if AdjointF is a boolean
        if not isinstance(AdjointF,bool):
            sys.exit('MarchenkoSeries: \'AdjointF\' must be a boolean.')
        
        # Select the set of Marchenko equation to be solved
        if AdjointF is True:
            a = 'a'
        else:
            a = ''
            
        # Model reflection response if not given    
        if R is None:
            R = self.RT_response_k1_w(normalisation=normalisation)['RP'+a]
        else:
            if np.shape(R) != (self.nf,self.nr):
                sys.exit('MarchenkoSeries: The reflection response \'R\' must '
                         +'have the shape (nf,nr).')
        
        # Model initial focusing function if not given    
        # The initial focusing function is multiplied by a k1-omega filter
        # and by a wavelet.
        if FP0 is None:
            FP0 = self.F_initial_k1_w(x3F=x3F,f0=f0,FK_filter=FK_filter,
                                      RelativeTaperLength=RelativeTaperLength,
                                      normalisation=normalisation,
                                      UpdateSelf=False)['FP0'+a]
            
        else:
            if np.shape(FP0) != (self.nf,self.nr):
                sys.exit('MarchenkoSeries: The initial focusing function '
                         +'\'FP0\' must have the shape (nf,nr).')
        
        # Determine the projector P if not given        
        if P is None:
            P = self.Projector_x1_t(x3F=x3F,f0=f0,delta=delta,
                                    RelativeTaperLength=RelativeTaperLength,
                                    UpdateSelf=False,
                                    normalisation=normalisation)['P'+a]
        else:
            if np.shape(P) != (self.nt,self.nr):
                sys.exit('MarchenkoSeries: The initial focusing function '
                         +'\'FP0\' must have the shape (nf,nr).')
      
        # Marchenko series
        FP = FP0.copy()
        FM = self.X1T2K1W(P*self.K1W2X1T(R*FP))

        for k in range(K):
            FP = FP0 + self.X1T2K1W( P*self.K1W2X1T(R*FM.conj()) ).conj()
            FM = self.X1T2K1W( P*self.K1W2X1T(R*FP) )
            
        # Update self
        if UpdateSelf is True:
            if x3F >= self.x3vec[-1]:
                _ = self.Insert_layer(x3=np.array([x3F,x3F+1]),
                                               UpdateSelf=UpdateSelf)
            else:
                _ = self.Insert_layer(x3=x3F,UpdateSelf=UpdateSelf)
        
            self.x3F = x3F
            
        out = {'FP':FP,'FM':FM,'FP0':FP0,'P':P,'R':R}
        return out
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routines for modelling wavefields in 1.5D non-reciprocal media.

.. module:: Wavefield_NRM_k1_w-v2.0

:Authors:
    Christian Reinicke (c.reinicke@tudelft.nl), Kees Wapenaar (), and Evert Slob ()
    
:Copyright:
    Christian Reinicke (c.reinicke@tudelft.nl), Kees Wapenaar (), and Evert Slob ()
"""

from Wavefield_NRM_k1_w import Wavefield_NRM_k1_w
import numpy as np
import sys

class Layered_NRM_k1_w(Wavefield_NRM_k1_w):
    """is a class to model wavefields in 1.5D (non-)reciprocal media in the 
    horizontal-wavenumber frequency domain.
        
    The class Layered_NRM_k1_w defines a 1.5D (non-)reciprocal medium and a 
    scalar wavefield. We consider all horizontal-wavenumbers and all 
    frequencies, that are sampled by the given number of samples and by the 
    given sample intervals, in space ('nr', 'dx1') as well as in time 
    ('nt', 'dt').

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
        Recommended value eps = :math:`\\frac{3 nf}{dt}`.
        
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
        scattering and propagation in the adjoint medium only for 
        flux-normalisation.
        
    Returns
    -------
    
    class
        A class to model a wavefield in a 1.5D non-reciprocal medium in the 
        horizontal-wavenumber frequency domain. The following instances are 
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
            
    
    .. todo:: 
        
        (1) In non-reciprocal media, when I use a complex-valued frequency 
        :math:`\omega'=\omega+\mathrm{j}\epsilon` the vertical wavenumber 
        definition becomes 
    
                :math:`k_3=\sqrt{(\\alpha \\beta -\gamma_1^2)\omega' 
                + 2\gamma_1 k_1 \omega' -k_1^2} = 
                \sqrt{(\\alpha \\beta - \gamma_1^2) (\omega -\epsilon^2) 
                + 2\gamma_1 k_1 \omega -k_1^2 
                + \mathrm{j} 2 \epsilon 
                [\omega (\\alpha \\beta -\gamma_1^2) + \gamma_1 k_1]}` .
        
            Hence, if 
              
                :math:`\mathrm{sign}(\epsilon 
                [\omega (\\alpha \\beta -\gamma_1^2) + \gamma_1 k_1]) < 0`, 
              
            the imaginary part of :math:`k_3` becomes negative, and the 
            wavefield components :math:`\mathrm{e}^{\mathrm{j}k_3 x_3}` become 
            unstable. I fix that by manually modifying the vertical wavenumber 
            to 
    
                :math:`k_{3,mod} = \sqrt{(\\alpha \\beta -\gamma_1^2) 
                (\omega - \epsilon^2) + 2\gamma_1 k_1 \omega -k_1^2 
                + \mathrm{j} 2 \Vert \epsilon 
                [\omega (\\alpha \\beta -\gamma_1^2) + \gamma_1 k_1] \Vert}` .
    
            For those :math:`\omega`-:math:`k_1` components for which the 
            absolute value in :math:`k_{3,mod}` has an effect, I effectively 
            change the sign of :math:`\epsilon`. Can we justify this fix? If 
            not, it might be better to simply exclude those 
            :math:`\omega`-:math:`k_1` components from the computation.
            
            In addition, the absolute value in :math:`k_{3,mod}` implies that 
            :math:`\epsilon` should be positive. Hence, when applying an 
            inverse Fourier transform to the time domain, one has to choose 
            the convention, 
            
                :math:`f(t)  = \int F(\omega) \; 
                \mathrm{e}^{\mathrm{j} \omega t} \mathrm{d}\omega`.
            
            This can be done by complex-conjugating the modelled wavefield 
            before applying the inverse Fourier transform. Note that the 
            default settings of the 2D inverse Fourier transform function 
            **K1W2X1T** handle all these sign choices correctly as long as 
            :math:`\epsilon >0`.
            
        (2) In reciprocal media, for Im(:math:`\gamma_i`) :math:`\\neq 0`, 
        energy conservation does not hold for evanescent waves.
            
        
    Notes
    -----
    
    - We format the data as described below.
        - Wavefields are saved in an array of dimensions (nf,nr) in the frequency domain and (nt,nr) in the time domain.
        - Wavefields are in the :math:`k_1`-:math:`\omega` domain.
            - The zero frequency component is placed at the first index position of the first dimension.
            - The zero horizontal-wavenumber component is placed at the first index position of the second dimension.
        - If the wavefield is transformed to the space-time domain: 
            - The zero time component is placed at the first index position of the first dimension, followed by nt/2-1 positive time samples and nt/2 negative time samples. 
            - The zero offset component is placed at the first index position of the second dimension, followed by nr/2-1 positive offset samples and nr/2 negative offset samples.
        - For evanescent waves, Kees makes a sign choice for the vertical-wavenumber,
        \t:math:`k_3' = -j \sqrt{k_1^2 - \omega^2 (\\alpha \\beta + \gamma_1^2 + \gamma_3^2)}`.
        - By default, **NumPy** makes the oppostie sign choice, 
        \t:math:`k_3' = +j \sqrt{k_1^2 - \omega^2 (\\alpha \\beta + \gamma_1^2 + \gamma_3^2)}`.
        - For convenience, we stick to **NumPy**'s sign choice. Thus, we will also adapt the sign choice for the propagators,
              - Kees chose: :math:`\\tilde{w}^{\pm} = \mathrm{exp}(-j k_3' \Delta x_3)`.
              - We choose: :math:`\\tilde{w}^{\pm} = \mathrm{exp}(+j k_3' \Delta x_3)`.
        
    References
    ----------
    
    Kees document as soon as it is published.
    
     
    Examples
    --------

    >>> from Layered_NRM_k1_w import Layered_NRM_k1_w as LM
    >>> import numpy as np
    
    >>> # Initialise wavefield in a layered non-reciprocal medium
    >>> F=LM(nt=1024, dt=0.005, nr=512, dx1=12.5 , 
    >>>      x3vec=np.array([1.1,2.2,3.7]), avec=np.array([1,2,3])*1e-3, 
    >>>      bvec=np.array([1.4,3.14,2])*1e-4, 
    >>>      g1vec=1j*np.array([0.8,2,1.3])*1e-4, 
    >>>      g3vec=1j*np.array([1.8,0.7,2.3])*1e-4, 
    >>>      ReciprocalMedium=True)  
    
    >>> # Get a meshgrid of the vertical-wavenumber
    >>> F.K3.shape
    (513, 512, 3)
    
    >>> # Get the vertical-wavenumber for omega=delta omega,  and k1=0
    >>> F.K3[1,0,0]
    (0.0003903913295999063+0j)
    
    **Wavefield Quantities**
    (Do not change the table!)
    
    +---------------------------+--------------------+--------------------+---------------+--------------------+
    |                           | *TE*               | *TM*               | *Ac. (fluid)* | *SH (solid)*       |
    +===========================+====================+====================+===============+====================+
    | **P**                     | :math:`E_2`        | :math:`H_2`        | :math:`p`     | :math:`v_2`        |
    +---------------------------+--------------------+--------------------+---------------+--------------------+
    | :math:`\mathbf{Q}_1`      | :math:`H_3`        | :math:`-E_3`       | :math:`v_1`   | :math:`-\\tau_{21}` |
    +---------------------------+--------------------+--------------------+---------------+--------------------+
    | :math:`\mathbf{Q}_3`      | :math:`-H_1`       | :math:`E_1`        | :math:`v_3`   | :math:`-\\tau_{23}` |
    +---------------------------+--------------------+--------------------+---------------+--------------------+
    | :math:`\mathbf{\\alpha}`   | :math:`\epsilon`   | :math:`\mu`        | :math:`\kappa`| :math:`\\rho`       |
    +---------------------------+--------------------+--------------------+---------------+--------------------+
    | :math:`\mathbf{\\beta}`    | :math:`\mu`        | :math:`\epsilon`   | :math:`\\rho`  | :math:`1/\mu`      |
    +---------------------------+--------------------+--------------------+---------------+--------------------+
    | :math:`\mathbf{\gamma}_1` | :math:`\\xi_{23}`   | :math:`-\zeta_{23}`| :math:`d_1`   | :math:`e_1`        |
    +---------------------------+--------------------+--------------------+---------------+--------------------+
    | :math:`\mathbf{\gamma}_3` | :math:`-\\xi_{21}`  | :math:`\zeta_{21}` | :math:`d_3`   | :math:`e_3`        |
    +---------------------------+--------------------+--------------------+---------------+--------------------+
    | :math:`\mathbf{\delta}_1` | :math:`\zeta_{32}` | :math:`-\\xi_{32}`  | :math:`e_1`   | :math:`d_1`        |
    +---------------------------+--------------------+--------------------+---------------+--------------------+
    | :math:`\mathbf{\delta}_3` | :math:`-\zeta_{12}`| :math:`\\xi_{12}`   | :math:`e_3`   | :math:`d_3`        |
    +---------------------------+--------------------+--------------------+---------------+--------------------+
    | :math:`\mathbf{B}`        | :math:`-J_2^e`     | :math:`-J_2^m`     | :math:`q`     | :math:`f_2`        |
    +---------------------------+--------------------+--------------------+---------------+--------------------+
    | :math:`\mathbf{C}_1`      | :math:`-J_3^m`     | :math:`J_3^e`      | :math:`f_1`   | :math:`2h_{21}`    |
    +---------------------------+--------------------+--------------------+---------------+--------------------+
    | :math:`\mathbf{C}_3`      | :math:`J_1^m`      | :math:`-J_1^e`     | :math:`f_3`   | :math:`2h_{23}`    |
    +---------------------------+--------------------+--------------------+---------------+--------------------+

    

   """    
  
    def __init__(self,nt,dt,nr,dx1,verbose=False,eps=None,x3vec=np.zeros(1),
                 avec=np.zeros(1),b11vec=np.zeros(1),b13vec=np.zeros(1),
                 b33vec=np.zeros(1),g1vec=np.zeros(1),g3vec=np.zeros(1),
                 ReciprocalMedium=False,AdjointMedium=False):
        
        # Inherit __init__ from Wavefield_NRM_k1_w
        Wavefield_NRM_k1_w.__init__(self,nt,dt,nr,dx1,verbose,eps)
        
        # Check if medium parameters are passed as arrays
        if not ( isinstance(x3vec,np.ndarray) and isinstance(avec,np.ndarray) 
        and isinstance(b11vec,np.ndarray) and isinstance(b13vec,np.ndarray) 
        and isinstance(b33vec,np.ndarray) and isinstance(g1vec,np.ndarray) 
        and isinstance(g3vec,np.ndarray) ):
            sys.exit('Layered_NRM_k1_w: x3vec, avec, b11vec, b13vec, b33vec, '
                     +'g1vec and g3vec have to be of the type numpy.ndarray.')
            
        # Set beta_{13}, gamma_1 and gamma_3 by default equal to zero
        if b13vec == np.zeros(1):
            b13vec = np.zeros_like(avec)
        if g1vec == np.zeros(1):
            g1vec = np.zeros_like(avec)
        if g3vec == np.zeros(1):
            g3vec = np.zeros_like(avec)
            
        # Force the medium parameters to have identical shape
        if (x3vec.shape!=avec.shape or x3vec.shape!=b11vec.shape 
            or x3vec.shape!=b13vec.shape or x3vec.shape!=b33vec.shape 
            or x3vec.shape!=g1vec.shape or x3vec.shape!=g3vec.shape):
            sys.exit('Layered_NRM_k1_w: x3vec, avec, b11vec, b13vec, b33vec, '
                     +'g1vec and g3vec have to be of identical shape.')
        
        # Force the medium parameters to be 1-dimensional, i.e. e.g. avec.shape=(n,)
        if x3vec.ndim!=1:
            sys.exit('Layered_NRM_k1_w: x3vec.ndim, avec.ndim, b11vec.ndim, '
                     +'b13vec.ndim, b33vec.ndim, g1vec.ndim and g3vec.ndim '
                     +'must be one.')
        
        # Check if x3vec is positive and constantly increasing
        if x3vec[0]<0 or (x3vec[1:]-x3vec[:-1] <= 0).any():
            sys.exit('Layered_NRM_k1_w: x3vec must only contain constantly '
                     +'increasing values greater than, or equal to zero.')
        
        # Check if Medium choices are bools
        if not (    isinstance(ReciprocalMedium,bool) 
                and isinstance(AdjointMedium,bool)   ):
            sys.exit('Layered_NRM_k1_w: \'ReciprocalMedium\' and '
                     +'\'AdjointMedium\' must be of the type bool.')
        
        # Check if medium parameters correspond to a lossless (non-)reciprocal medium
        if ReciprocalMedium == False:
            if (avec.imag.any()!=0      or b11vec.imag.any()!=0 
                or b13vec.imag.any()!=0 or b33vec.imag.any()!=0 
                or g1vec.imag.any()!=0  or g3vec.imag.any()!=0 ):
                sys.exit('Layered_NRM_k1_w: In lossless non-reciprocal media '
                         'avec, b11vec, b13vec, b33vec, g1vec and g3vec have '
                         +'to be real-valued.')
        # The updated equations do not address reciprocal media. Therefore,
        # I am not making any assumptions about their real- and imaginary
        # parts before Kees confirms the missing expressions.
        elif ReciprocalMedium == True:
            sys.exit('Layered_NRM_k1_w: This version (2.0) does not yet '
                     +'include expressions for reciprocal media. To consider '
                     +'reciprocal media set \'ReciprocalMedium=False\' and '
                     +'\'g1vec = 0\' and \'g3vec = 0\'.')
#            if (avec.imag.any()!=0      or b11vec.imag.any()!=0 
#                or b13vec.imag.any()!=0 or b33vec.imag.any()!=0 
#                or g1vec.real.any()!=0  or g3vec.real.any()!=0 ):
#                sys.exit('Layered_NRM_k1_w: In lossless reciprocal media avec,'
#                         +' b11vec, b13vec and b33vec have to be real-valued, '
#                         +'g1vec and g3vec have to be only imaginary-valued.')
            
        # Set medium parameters 
        self.x3vec  = x3vec
        self.avec   = avec
        self.b11vec = b11vec
        self.b13vec = b13vec
        self.b31vec = b13vec
        self.b33vec = b33vec
        self.g1vec  = g1vec
        self.g3vec  = g3vec
        self.ReciprocalMedium = ReciprocalMedium
        self.AdjointMedium    = AdjointMedium
        
        # Calculate vertical ray-parameter p3=p3(+p1) p3n=p3(-p1) 
        # Note: By default python uses opposite sign convention for evanescent waves as Kees: (-1)**0.5=1j
        K3  = np.zeros((self.nf,self.nr,self.x3vec.size),dtype=complex)
        K3n = K3.copy()
        W   = self.W_K1_grid()['Wgrid']
        K1  = self.W_K1_grid()['K1gridfft']
        if self.ReciprocalMedium is True:
            tmp = self.avec*self.bvec + self.g1vec**2 + self.g3vec**2
            for layer in range(self.x3vec.size):
                K3[:,:,layer] = tmp[layer]*W**2 - K1**2
            K3n = K3.copy()
        elif self.ReciprocalMedium is False:
            tmp = self.avec*self.bvec - self.g1vec**2
            if self.eps is None:
                for layer in range(self.x3vec.size):
                    K3[:,:,layer]  = tmp[layer]*W**2 + 2*self.g1vec[layer]*K1*W - K1**2
                    K3n[:,:,layer] = tmp[layer]*W**2 - 2*self.g1vec[layer]*K1*W - K1**2
                    
            # This is a manual fix to avoid exponential growth of the wavefield
            # I believe this fix sign-inverts epsilon for the unstable
            # w-k1 components. I am not sure if that is a problem. 
            else:
                for layer in range(self.x3vec.size):
                    K3[:,:,layer]  = (tmp[layer]*W**2 + 2*self.g1vec[layer]*K1*W - K1**2)
                    K3n[:,:,layer] = (tmp[layer]*W**2 - 2*self.g1vec[layer]*K1*W - K1**2)
                    
                    K3[:,:,layer]  = K3[:,:,layer].real  + 1j*np.abs(K3[:,:,layer].imag)
                    K3n[:,:,layer] = K3n[:,:,layer].real + 1j*np.abs(K3n[:,:,layer].imag)
                    
        self.K3  = K3**0.5
        self.K3n = K3n**0.5
        
    
    def FK1_mask_k1_w(self,RelativeTaperLength=2**(-5),wmax=None,Opening=1.0):
        """returns a mask that mutes evanescent waves in the :math:`k_1`-
        :math:`\omega` domain.
        
        Parameters
        ----------
            
        RelativeTaperLength : int, float, optional
            The product of \'RelativeTaperLength\' and the number of temporal 
            samples \'nt\' determines the taper length. The default value is 
            \'RelativeTaperLength\':math:`=2^{-5}.`
            
        wmax : int, float, complex, optional
            Cut-off frequency :math:`\omega_{\mathrm{max}}` in :math:`s^{-1}`.
            
        Opening : int, float, optional
            Factor, greater than zero, to widen or tighten the :math:`\omega`-
            :math:`k_1` mask. If equal to one, the opening corresponds to the 
            transition from propagating to evanescent waves. 
            
        Returns
        -------
        
        dict
            A dictionary that contains the ,
                - **FK**: Sharp-edged :math:`\omega`-:math:`k_1` mask, mutes 
                the evanescent wavefield in each layer indivually. Shape 
                (nf,nr,n).
                
                - **FK_global**: Sharp-edged :math:`\omega`-:math:`k_1` mask, 
                mutes the evanescent wavefield in the entire model. Shape 
                (nf,nr).
                
                - **FK_tap**: Tapered :math:`\omega`-:math:`k_1` mask, mutes 
                the evanescent wavefield in each layer indivually. Shape 
                (nf,nr,n).
                
                - **FK_global_tap**: Tapered :math:`\omega`-:math:`k_1` mask, 
                mutes the evanescent wavefield in the entire model. Shape 
                (nf,nr).
                
                - **FKn**: Sharp-edged :math:`\omega`-:math:`k_1` mask, mutes 
                the evanescent wavefield in each layer indivually (for sign-
                inverted :math:`k_1`). Shape (nf,nr,n).
                
                - **FKn_global**: Sharp-edged :math:`\omega`-:math:`k_1` mask, 
                mutes the evanescent wavefield in the entire model (for sign-
                inverted :math:`k_1`). Shape (nf,nr).
                
                - **FKn_tap**: Tapered :math:`\omega`-:math:`k_1` mask, mutes 
                the evanescent wavefield in each layer indivually (for sign-
                inverted :math:`k_1`). Shape (nf,nr,n).
                
                - **FKn_global_tap**: Tapered :math:`\omega`-:math:`k_1` mask,
                mutes the evanescent wavefield in the entire model (for sign-
                inverted :math:`k_1`). Shape (nf,nr).
                
                - **taperlen**: Taper length in number of samples.
                
            All masks are stored as complex valued arrays because they will be 
            applied to complex-valued arrays.
            
        Examples
        --------
        
        >>> # Initialise wavefield
        >>> from Layered_NRM_k1_w import Layered_NRM_k1_w as LM
        >>> F=LM(nt=1024,dt=0.005,nr=512,dx1=12.5,x3vec=np.array([1.1,2.2,3.7]),
        >>>      avec=np.array([1,2,3])*1e-3,bvec=np.array([1.4,3.14,2])*1e-4,
        >>>      g1vec=1j*np.array([0.8,2,1.3])*1e-4,
        >>>      g3vec=1j*np.array([1.8,0.7,2.3])*1e-4,
        >>>      ReciprocalMedium=True)
        
        >>> # Create fk mask with a cut-off frequency at 200 1/s
        >>> Mask=F.FK1_mask2_k1_w(wmax=200)
        >>> Tapered_fk_mask = Mask['FK_tap']
        
            
        """
        # Check if RelativeTaperLength is a float or an int
        if not ( isinstance(RelativeTaperLength,int) 
              or isinstance(RelativeTaperLength,float) ):
            sys.exit('FK1_mask2_k1_w: \'RelativeTaperLength\' must be of the '
                     +'type int or float.')
            
        # Check that RelativeTaperLength is not smaller than zero
        if RelativeTaperLength < 0:
            sys.exit('FK1_mask2_k1_w: \'RelativeTaperLength\' must be greater '
                     +'than, or equal to zero.')
            
        # Check if wmax is a float or an int or a complex
        if wmax is not None:
            if not ( isinstance(wmax,int) 
                  or isinstance(wmax,float) 
                  or isinstance(wmax,complex)):
                sys.exit('FK1_mask2_k1_w: \'wmax\' must be of the type int, '
                         +'float or complex.')
       
            # Check that wmax is not smaller than zero
            if wmax.real < 0:
                sys.exit('FK1_mask2_k1_w: \'wmax\' must be greater than, or '
                         +'equal to zero.')
                
        # Check if Opening is a scalar
        if not ( isinstance(Opening,int) 
              or isinstance(Opening,float) ):
            sys.exit('FK1_mask2_k1_w: \'Opening\' must be of the '
                     +'type int or float.')
            
        # Check that Opening is not smaller than zero
        if Opening < 0:
            sys.exit('FK1_mask2_k1_w: \'Opening\' must be greater '
                     +'than, or equal to zero.')
            
        # Help variable (k3 squared)
        Q  = np.ones((self.nf,self.nr,self.x3vec.size),dtype=complex)
        Qn = np.ones((self.nf,self.nr,self.x3vec.size),dtype=complex)
            
        # Sharp-edged FK mask
        FK  = np.ones((self.nf,self.nr,self.x3vec.size),dtype=complex)
        FKn = np.ones((self.nf,self.nr,self.x3vec.size),dtype=complex)
        
        # Frequency and wavenumber meshgrid
        W  = self.W_K1_grid()['Wgrid'].real
        K1 = self.W_K1_grid()['K1gridfft']
        
        if self.ReciprocalMedium is True:
            tmp = self.avec*self.bvec + self.g1vec**2 + self.g3vec**2
            for layer in range(self.x3vec.size):
                Q[:,:,layer] = Opening**2*tmp[layer]*W**2 - K1**2
            Qn = Q.copy()
        else:
            tmp = self.avec*self.bvec - self.g1vec**2
            for layer in range(self.x3vec.size):
                Q[:,:,layer]  = ( Opening**2*tmp[layer]*W**2 
                                 + 2*self.g1vec[layer]*K1*W - K1**2 )
                Qn[:,:,layer] = ( Opening**2*tmp[layer]*W**2 
                                 - 2*self.g1vec[layer]*K1*W - K1**2 )
                    
        FK[Q   <= 0] = 0
        FKn[Qn <= 0] = 0
        FK_global  = np.prod(FK,-1)
        FKn_global = np.prod(FKn,-1)
        
        FK_tap         = FK.copy()
        FKn_tap        = FKn.copy()
        FK_global_tap  = FK_global.copy()
        FKn_global_tap = FKn_global.copy()
        
        # Taper length
        taperlen = int(RelativeTaperLength*self.nt)
        
        if taperlen != 0:

            # Define sine taper
            # If complex-valued frequencies are used they must be included in
            # the taper. Otherwise the amplitudes are falsified.
            w  = self.Wvec()[:,0]
            def SineTaper(start,ind):
                w0 = np.real(w[ind])
                tap = np.sin( (w[start:start+taperlen]-w0)
                              /(self.Dw()*(taperlen-1)  )
                              *np.pi*0.5)
                return tap**2
                        
            # Define taper for each wavenumber
            for WavNum in range(self.nr):
                
                # FK_global
                i1 = np.where(FK_global[:,WavNum]==1)[0]
                if i1.size != 0:
                    i1  = i1[0]    
                
                    if i1 + taperlen <= self.nf:
                        FK_global_tap[i1:i1+taperlen,WavNum] = SineTaper(i1,i1)
                      
                    else:
                        FK_global_tap[i1:,WavNum] = SineTaper(i1,i1)[:self.nf-i1]
                   
                # FKn_global
                i1 = np.where(FKn_global[:,WavNum]==1)[0]
                if i1.size != 0:
                    i1  = i1[0]    
                
                    if i1 + taperlen <= self.nf:
                        FKn_global_tap[i1:i1+taperlen,WavNum] = SineTaper(i1,i1)
                       
                    else:
                        FKn_global_tap[i1:,WavNum] = SineTaper(i1,i1)[:self.nf-i1]
                
                for layer in range(self.x3vec.size):
                    # FK
                    i1 = np.where(FK[:,WavNum,layer]==1)[0]
                    if i1.size != 0:
                        i1  = i1[0]    
                    
                        if i1 + taperlen <= self.nf:
                            FK_tap[i1:i1+taperlen,WavNum,layer] = SineTaper(i1,i1)
                        
                        else:
                            FK_tap[i1:,WavNum,layer] = SineTaper(i1,i1)[:self.nf-i1]
                            
                    # FKn
                    i1 = np.where(FKn[:,WavNum,layer]==1)[0]
                    if i1.size != 0:
                        i1  = i1[0]    
                    
                        if i1 + taperlen <= self.nf:
                            FKn_tap[i1:i1+taperlen,WavNum,layer] = SineTaper(i1,i1)
                        
                        else:
                            FKn_tap[i1:,WavNum,layer] = SineTaper(i1,i1)[:self.nf-i1]
        
        # Mask to cut-off frequencies greater than wmax
        if wmax is not None:
            ind = int(np.ceil(wmax.real/self.Dw()))
            
            if ind > 0:
                M   = np.ones((self.nf,self.nr,1),dtype=complex)
                M[ind:,:,0] = 0
                M = np.repeat(M,FK.shape[-1],axis=2)
                FK         = M*FK
                FK_global  = M[:,:,0] * FK_global
                FKn        = M*FKn
                FKn_global = M[:,:,0] * FKn_global
                
                if taperlen < ind:
                    M[ind-taperlen:ind,0,0] = SineTaper(ind-taperlen,ind)
                    M[:,:,0]   = np.repeat(M[:,:1,0],FK.shape[1],axis=1)
                    M          = np.repeat(M[:,:,:1],FK.shape[2],axis=2)
                FK_tap         = M*FK_tap
                FK_global_tap  = M[:,:,0] * FK_global_tap
                FKn_tap        = M*FKn_tap
                FKn_global_tap = M[:,:,0] * FKn_global_tap
            else:
                FK        = 0*FK
                FK_global = 0*FK_global
                FK_tap        = 0*FK_tap
                FK_global_tap = 0*FK_global_tap
                
                FKn        = 0*FKn
                FKn_global = 0*FKn_global
                FKn_tap        = 0*FKn_tap
                FKn_global_tap = 0*FKn_global_tap
        
        out = {'FK':FK,'FK_global':FK_global,'FK_tap':FK_tap,
               'FK_global_tap':FK_global_tap,
               'FKn':FKn,'FKn_global':FKn_global,'FKn_tap':FKn_tap,
               'FKn_global_tap':FKn_global_tap,'taperlen':taperlen}
        return out
        
    def L_eigenvectors_k1_w(self,beta=None,g3=None,K3=None,K3n=None,normalisation='flux'):
        """computes the eigenvector matrix 'L' and its inverse 'Linv', either in flux- or in pressure-normalisation for the vertical-wavenumber 'K3' inside a homogeneous layer. Here, the vertical-wavenumber is a meshgrid that contains all combinations of frequencies :math:`\omega` and horizontal-wavenumbers :math:`k_1`. If \'AdjointMedium=True\', **L_eigenvectors_k1_w** also computes the eigenvector matrix in the adjoint medium 'La' and its inverse 'Lainv'. 
        
        Parameters
        ----------
    
        beta : int, float
            Medium parameter :math:`\\beta`  (real-valued).
        
        g3 : int, float
            Medium parameter :math:`\gamma_3`.
        
        K3 : numpy.ndarray
            Vertical-wavenumber :math:`k_3(+k_1)` for all frequencies :math:`\omega` and horizontal-wavenumbers :math:`k_1`.
        
        K3n : numpy.ndarray, optional (required if 'AdjointMedium=True')
            Vertical-wavenumber :math:`k_3(-k_1)` for all frequencies :math:`\omega` and sign-inverted horizontal-wavenumbers :math:`k_1`.
            
        normalisation : str, optional
            For pressure-normalisation set normalisation='pressure', for flux-normalisation set normalisation='flux'.
            
        Returns
        -------
    
        dict
            Dictionary that contains 
                - **L**: The eigenvector matrix.
                - **Linv**: The inverse of the eigenvector matrix.
                - **La**: The eigenvector matrix (adjoint medium).
                - **Lainv**: The inverse of the eigenvector matrix (adjoint medium).
            All eigenvector matrices are stored in a in a (nf,nr,2,2)-array. The first two dimensions correspond to all combinations of frequencies :math:`\omega` and horizontal-wavenumbers :math:`k_1`. The last two dimension are the actual eigenvector matrices for all :math:`\omega`-:math:`k_1` components.
        
        Notes
        -----
            - The eigenvector matrix 'L' and its inverse 'Linv' are different for reciprocal and non-reciprocal media.
            - For reciprocal media, the eigenvectors of the adjoint medium are identical to the eigenvectors of the true medium.
            - We have defined the eigenvectors of the adjoint medium only for flux-normalisation.
            - At zero frequency (:math:`\omega=0 \;\mathrm{s}^{-1}`), the eigenvector matrices \'L\' and their inverse \'Linv\' contain elements with poles. For computational convenience, we set the poles equal to zero. However, the resulting zero frequency component of all outputs is meaningless.
        
        References
        ----------
        Kees document as soon as it is published.

        Examples
        --------

        >>> from Layered_NRM_k1_w import Layered_NRM_k1_w as LM
        >>> import numpy as np
        
        >>> # Initialise wavefield in a layered non-reciprocal medium
        >>> F=LM(nt=1024,dt=0.005,nr=512,dx1=12.5,x3vec=np.array([1.1,2.2,3.7]),
        >>>      avec=np.array([1,2,3])*1e-3,bvec=np.array([1.4,3.14,2])*1e-4,
        >>>      g1vec=1j*np.array([0.8,2,1.3])*1e-4,
        >>>      g3vec=1j*np.array([1.8,0.7,2.3])*1e-4,
        >>>      ReciprocalMedium=True)
        
        >>> # Compute eigenvectors in pressure-normalisation
        >>> Leig=F.L_eigenvectors_k1_w(F.bvec[0],F.g3vec[0],
        >>>                            F.K3[:,:,0],normalisation='pressure')
        >>> L=Leig['L']
        
        >>> # For pressure normalisation, the top-left element of L equals
        >>> # 1 for all frequencies and all horizontal-wavenumbers
        >>> L[101,200,0,0]
        (1+0j)
        
        >>> # For pressure normalisation, the bottom-left element of L does not
        >>> # equal 1 for all frequencies and all horizontal-wavenumbers
        >>> L[101,200,1,0]
        12.370632073930679j
        
        """
        # Check if required input variables are given
        if (beta is None) or (g3 is None) or (K3 is None):
            sys.exit('L_eigenvectors_k1_w: The input variables \'beta\', \'g3\' and  \'K3\' must be set.')
         
        # Check if normalisation is set correctly
        if (normalisation is not 'flux') and (normalisation is not 'pressure'):
            sys.exit('L_eigenvectors_k1_w: The input variable \'normalisation\' must be set, either to \'flux\', or to \'pressure\'.')
            
        # Check if the vertical-wavenumber for a negative horizontal-wavenumber is given
        if  (self.AdjointMedium is True) and (K3n is None) and (normalisation is 'flux') and (self.ReciprocalMedium is False):
            sys.exit('L_eigenvectors_k1_w: The input variable \'K3n\' (vertical-wavenumber K3 for a negative horizontal-wavenumber k1) must be given to compute the eigenvector matrix of the adjoint medium \'La\' and its inverse\'Lainv\'.')
            
        # Initialise L and Linv
        L = np.zeros((self.nf,self.nr,2,2),dtype=complex)
        Linv = np.zeros((self.nf,self.nr,2,2),dtype=complex)
        La = None
        Lainv = None
        
        # Construct a vertical ray-parameter
        # Exclude poles at zero-frequency
        Om           = self.W_K1_grid()['Wgrid']
        P3           = K3.copy()
        P3[1:,:]     = K3[1:,:]/Om[1:,:]
        P3[0,:]      = 0
        
        # Construct an inverse vertical ray-parameter
        # Exclude pole at zero frequency and zero horizontal-wavenumber
        # K3[0,0] = 0
        # P3[0,:] = 0
        P3inv = P3.copy()
        P3inv[1:,:] = Om[1:,:]/K3[1:,:]
        
        if self.verbose is True:
            print('\nL_eigenvectors_k1_w ')
            print('-------------------------------------------------------------------------------')
            print('The eigenvectors L and their inverse Linv have a pole at zero frequency. Here, ')
            print('we set the zero frequency component of L and Linv to zero (which is wrong but ')
            print('convenient for the computation).\n')

            
        if (self.ReciprocalMedium is True) and (normalisation is 'flux'):
            # L matrix
            fac = (beta*P3inv/2)**0.5
            L[:,:,0,0] = 1*fac
            L[:,:,0,1] = 1*fac
            L[:,:,1,0] = (P3+g3)/beta*fac
            L[:,:,1,1] = -(P3-g3)/beta*fac
            
            # Inverse L matrix
            Linv[:,:,0,0] = -L[:,:,1,1]
            Linv[:,:,0,1] =  L[:,:,0,0]
            Linv[:,:,1,0] =  L[:,:,1,0]
            Linv[:,:,1,1] = -L[:,:,0,0]
            
            if self.AdjointMedium is True:
                if self.verbose is True:
                    print('\nL_eigenvectors_k1_w (AdjointMedium is True) and (ReciprocalMedium is True)')
                    print('-------------------------------------------------------------------------------')
                    print('For reciprocal media, the eigenvector matrix of a medium and its adjoint medium ')
                    print('are identical.\n')
               
                La = L.copy()
                Lainv = Linv.copy()
            
        elif (self.ReciprocalMedium is True) and (normalisation is 'pressure'):            
            # L matrix
            L[:,:,0,0] = 1
            L[:,:,0,1] = 1
            L[:,:,1,0] = (P3+g3)/beta
            L[:,:,1,1] = -(P3-g3)/beta
            
            # Inverse L matrix
            fac = beta*P3inv/2
            Linv[:,:,0,0] = -L[:,:,1,1]*fac
            Linv[:,:,0,1] = 1*fac
            Linv[:,:,1,0] = L[:,:,1,0]*fac
            Linv[:,:,1,1] = -1*fac
            
            if self.verbose is True and self.AdjointMedium is True:
                print('\nL_eigenvectors_k1_w (AdjointMedium is True) and (normalisation=\'pressure\')')
                print('-------------------------------------------------------------------------------')
                print('We have defined the eigenvector matrix of the adjoint medium \'La\' and its ')
                print('inverse \'Lainv\' only for flux-normalisation.\n')
            
        elif (self.ReciprocalMedium is False) and (normalisation is 'flux'):
            # L matrix
            L[:,:,0,0] = (beta*P3inv)**0.5
            L[:,:,0,1] = L[:,:,0,0]
            L[:,:,1,0] = (P3/beta)**0.5
            L[:,:,1,1] = -L[:,:,1,0]
            L = L/2**0.5
            
            # Inverse L matrix
            Linv[:,:,0,0] = L[:,:,1,0]
            Linv[:,:,0,1] = L[:,:,0,0]
            Linv[:,:,1,0] = L[:,:,1,0]
            Linv[:,:,1,1] = -L[:,:,0,0]
            
            if self.AdjointMedium is True:
                if self.verbose is True:
                    print('\nL_eigenvectors_k1_w (AdjointMedium is True) and (ReciprocalMedium is False)')
                    print('-------------------------------------------------------------------------------')
                    print('For non-reciprocal media, the eigenvector matrix of a medium and its adjoint ')
                    print('medium are different.\n')
                    
                # Construct a vertical ray-parameter  for sign-inverted 
                # horizontal-wavenumbers
                # Exclude poles at zero-frequency
                Om           = self.W_K1_grid()['Wgrid']
                P3n           = K3n.copy()
                P3n[1:,:]     = K3n[1:,:]/Om[1:,:]
                P3n[0,:]      = 0
                
                # Construct an inverse vertical ray-parameter  for sign-inverted 
                # horizontal-wavenumbers
                # Exclude pole at zero frequency and zero horizontal-wavenumber
                # K3[0,0] = 0
                # P3[0,:] = 0
                P3ninv = P3n.copy()
                P3ninv[1:,:] = Om[1:,:]/K3n[1:,:]
                
                # L matrix (adjoint medium) = N Transpose( Inverse( L(-k1) )) N
                La = np.zeros((self.nf,self.nr,2,2),dtype=complex)
                La[:,:,0,0] = (beta*P3ninv)**0.5
                La[:,:,0,1] = La[:,:,0,0]
                La[:,:,1,0] = (P3n/beta)**0.5
                La[:,:,1,1] = -La[:,:,1,0]
                La = La/2**0.5
                
                #  Inverse L matrix (adjoint medium) = N Transpose( L(-k1)) N
                Lainv = np.zeros((self.nf,self.nr,2,2),dtype=complex)
                Lainv[:,:,0,0] = (P3n/beta)**0.5
                Lainv[:,:,0,1] = (beta*P3ninv)**0.5
                Lainv[:,:,1,0] = Lainv[:,:,0,0]
                Lainv[:,:,1,1] = -Lainv[:,:,0,1]
                Lainv = Lainv/2**0.5
            
        elif (self.ReciprocalMedium is False) and (normalisation is 'pressure'):
            # L matrix
            L[:,:,0,0] = 1
            L[:,:,0,1] = 1
            L[:,:,1,0] = P3/beta
            L[:,:,1,1] = -P3/beta
            
            # Inverse L matrix
            Linv[:,:,0,0] = 1
            Linv[:,:,0,1] = beta*P3inv
            Linv[:,:,1,0] = 1
            Linv[:,:,1,1] = -beta*P3inv
            Linv = 0.5*Linv
            
            if self.verbose is True and self.AdjointMedium is True:
                print('\nL_eigenvectors_k1_w (AdjointMedium is True) and (normalisation=\'pressure\')')
                print('-------------------------------------------------------------------------------')
                print('We have defined the eigenvector matrix of the adjoint medium \'La\' and its ')
                print('inverse \'Lainv\' only for flux-normalisation.\n')
        
        out = {'L':L,'Linv':Linv,'La':La,'Lainv':Lainv}
        return out
          
    def RT_k1_w(self,beta_u=None,g3_u=None,K3_u=None,K3n_u=None,
                     beta_l=None,g3_l=None,K3_l=None,K3n_l=None,
                     normalisation='flux'):
        """computes the scattering coefficients at an horizontal interface.
        
        The scattering coefficients can be computed either in flux- or in pressure-normalisation. The variables with subscript 'u' refer to the medium parameters in the upper half-space, the variables with subscript 'l' refer to the medium parameters in the lower half-space. The vertical-wavenumbers \'K3\':math:`=k_3(k_1,\omega)` and \'K3n\':math:`=k_3(-k_1,\omega)` are stored as :math:`k_1`-:math:`\omega` meshgirds to compute the scattering coefficients for all sampled frequencies and horizontal-wavenumbers in a vectorsied manner. Set \'AdjointMedium=True\' to compute the scattering coefficients also in the adjoint medium.
        
        Parameters
        ----------
    
        beta_u : int, float
            Medium parameter :math:`\\beta` (real-valued) (upper half-space).
        
        g3_u : int, float
            Medium parameter :math:`\gamma_3` (upper half-space).
        
        K3_u : numpy.ndarray
            Vertical-wavenumber :math:`k_{3,u}(+k_1)` for all frequencies :math:`\omega` and horizontal-wavenumbers :math:`k_1` (upper half-space).
        
        K3n_u : numpy.ndarray, optional (required if 'AdjointMedium=True')
            Vertical-wavenumber :math:`k_{3,u}(-k_1)` for all frequencies :math:`\omega` and sign-inverted horizontal-wavenumbers :math:`k_1` (upper half-space).
            
        beta_l : int, float
            Medium parameter :math:`\\beta` (real-valued) (lower half-space).
        
        g3_l : int, float
            Medium parameter :math:`\gamma_3` (lower half-space).
        
        K3_l : numpy.ndarray
            Vertical-wavenumber :math:`k_{3,l}(+k_1)` for all frequencies :math:`\omega` and horizontal-wavenumbers :math:`k_1` (lower half-space).
        
        K3n_l : numpy.ndarray, optional (required if 'AdjointMedium=True')
            Vertical-wavenumber :math:`k_{3,l}(-k_1)` for all frequencies :math:`\omega` and sign-inverted horizontal-wavenumbers :math:`k_1` (lower half-space).
            
        normalisation : str, optional
            For pressure-normalisation set normalisation='pressure', for flux-normalisation set normalisation='flux'.
            
        Returns
        -------
    
        dict
            Dictionary that contains 
                - **rP**: Reflection coefficient from above.
                - **tP**: Transmission coefficient from above 'tP'.
                - **rM**: Reflection coefficient from below.
                - **tM**: Transmission coefficient from below.
                - **rPa**: Reflection coefficient from above (adjoint medium).
                - **tPa**: Transmission coefficient from above (adjoint medium).
                - **rMa**: Reflection coefficient from below (adjoint medium).
                - **tMa**: Transmission coefficient from below (adjoint medium).
            All scattering coefficients are stored as arrays with the shape (nf,nr).
            
        .. todo:: 
        
            (1) For :math:`k_1,\omega=(0,0)` there is a zero division in the 
            computation of the scattering coefficients. I have fixed that, 
            however, I believe that the fix is (mathmatically) wrong. Check that!
        
        Notes
        -----
    
        - For reciprocal media, the scattering coefficients of the adjoint medium are identical to the scattering coefficients of the true medium.
        - We have defined the scattering coefficients of the adjoint medium only for flux-normalisation.
        
            
        References
        ----------
        Kees document as soon as it is published.
        
        Examples
        --------
    
        >>> from Layered_NRM_k1_w import Layered_NRM_k1_w as LM
        >>> import numpy as np
        
        >>> # Initialise wavefield F in a reciprocal medium 
        >>> F=LM(nt=1024,dt=0.005,nr=512,dx1=12.5,x3vec=np.array([1.1,2.2,3.7]),
        >>>      avec=np.array([1,2,3])*1e-3,bvec=np.array([1.4,3.14,2])*1e-4,
        >>>      g1vec=1j*np.array([0.8,2,1.3])*1e-4,
        >>>      g3vec=1j*np.array([1.8,0.7,2.3])*1e-4,
        >>>      ReciprocalMedium=True)
        
        >>> # Compute scattering coefficients at the first interface in flux
        >>> # normalisation
        >>> Scat=F.RT_k1_w(beta_u=F.bvec[0],g3_u=F.g3vec[0],K3_u=F.K3[:,:,0],
        >>>                K3n_u=F.K3n[:,:,0],
        >>>                beta_l=F.bvec[1],g3_l=F.g3vec[1],K3_l=F.K3[:,:,1],
        >>>                K3n_l=F.K3n[:,:,1],normalisation='flux')
        
        >>> # Read the scattering coeffcients, and 
        >>> tP = Scat['tP']
        >>> rM = Scat['rM']
        >>> rP = Scat['rP']
        >>> tM = Scat['tM'] 
        
        >>> tP.shape
        (513, 512)
        
        >>> np.linalg.norm(tP-tM)
        0.0
        
        
        """
        
        # Check if required input variables are given
        if ((beta_u is None) or (g3_u is None) or (K3_u is None) 
         or (beta_l is None) or (g3_l is None) or (K3_l is None)):
            sys.exit('RT_k1_w: The input variables \'beta_u\', \'g3_u\',  '+
                     '\'K3_u\', \'beta_l\', \'g3_l\',  \'K3_l\' must be set.')
            
        # Check if normalisation is set correctly
        if (normalisation is not 'flux') and (normalisation is not 'pressure'):
            sys.exit('RT_k1_w: The input variable \'normalisation\' must be'+
                     ' set, either to \'flux\', or to \'pressure\'.')
            
        # Check if the vertical-wavenumber for a sign-inverted horizontal-
        # wavenumber is given
        if  (    (self.AdjointMedium is True) 
             and ((K3n_u is None) or (K3n_l is None)) 
             and (normalisation is 'flux') 
             and (self.ReciprocalMedium is False)):
            sys.exit('RT_k1_w: The input variables \'K3n_u\' and \'K3n_l\''+
                     ' (vertical-wavenumber :math:`k_3` for a sign-inverted'+
                     ' horizontal-wavenumber :math:`k_1`) must be set to'+
                     ' compute the scattering coefficients in the adjoint'+
                     ' medium \'rPa\', \'tPa\', \'rMa\' and \'tMa\'.')
            
        # Initialise scattering coefficients in adjoint medium    
        rPa = None
        tPa = None
        rMa = None
        tMa = None
            
        # Frequency meshgrid
        Om = self.W_K1_grid()['Wgrid']
        
        # For zero frequency and zero horizontal-wavenumber we will encounter
        # divisions by zero. 
        # To avoid this problem we modify the (w,k1)=(0,0) element of K3,
        # such that there is no division by zero, and such that the resulting
        # scattering coefficients are correct
        # The (w,k1)=(0,0) element of K3 is actually a ray-parameter P3
        K3_u[0,0]  = K3_u[1,0]/self.Dw()
        K3_l[0,0]  = K3_l[1,0]/self.Dw()
        K3n_u[0,0] = K3n_u[1,0]/self.Dw()
        K3n_l[0,0] = K3n_l[1,0]/self.Dw()
        
        if (self.ReciprocalMedium is True) and (normalisation is 'flux'):
            
            # True medium
            denom = 1/( (K3_u-Om*g3_u)*beta_l + (K3_l+Om*g3_l)*beta_u )
            rP =  ( (K3_u+Om*g3_u)*beta_l - (K3_l+Om*g3_l)*beta_u ) * denom 
            rM = -( (K3_u-Om*g3_u)*beta_l - (K3_l-Om*g3_l)*beta_u ) * denom
            tP = 2*(K3_u*beta_l*K3_l*beta_u)**0.5 * denom
            tM = tP
            
            # Correct the zero frequency, zero horizontal-wavenumber component
            denom = 1/( (K3_u[0,0]-g3_u)*beta_l + (K3_l[0,0]+g3_l)*beta_u )
            rP[0,0] =  ( (K3_u[0,0]+g3_u)*beta_l - (K3_l[0,0]+g3_l)*beta_u ) * denom 
            rM[0,0] = -( (K3_u[0,0]-g3_u)*beta_l - (K3_l[0,0]-g3_l)*beta_u ) * denom
            tP[0,0] = 2*(K3_u[0,0]*beta_l*K3_l[0,0]*beta_u)**0.5 * denom
            tM[0,0] = tP[0,0]
            
            if self.AdjointMedium is True:
                # Adjoint medium
                rPa = rP 
                tPa = tP 
                rMa = rM 
                tMa = tM 
    
                if self.verbose is True:
                    print('\nRT_k1_w: (AdjointMedium is True) and (ReciprocalMedium is True)')
                    print(72*'-')
                    print('For reciprocal media, the scattering coefficients in a medium and its adjoint medium are identical.\n')
        
        elif (self.ReciprocalMedium is True) and (normalisation is 'pressure'):
            
            # True medium
            denom = 1/( (K3_u-Om*g3_u)*beta_l + (K3_l+Om*g3_l)*beta_u )
            rP =  ( (K3_u+Om*g3_u)*beta_l - (K3_l+Om*g3_l)*beta_u ) * denom
            rM = -( (K3_u-Om*g3_u)*beta_l - (K3_l-Om*g3_l)*beta_u ) * denom
            tP = 2*K3_u*beta_l * denom
            tM = 2*K3_l*beta_u * denom
            
            # Correct the zero frequency, zero horizontal-wavenumber component
            denom = 1/( (K3_u[0,0]-g3_u)*beta_l + (K3_l[0,0]+g3_l)*beta_u )
            rP[0,0] =  ( (K3_u[0,0]+g3_u)*beta_l - (K3_l[0,0]+g3_l)*beta_u ) * denom 
            rM[0,0] = -( (K3_u[0,0]-g3_u)*beta_l - (K3_l[0,0]-g3_l)*beta_u ) * denom
            tP[0,0] = 2*K3_u[0,0]*beta_l * denom
            tM[0,0] = 2*K3_l[0,0]*beta_u * denom
            
            if self.verbose is True and self.AdjointMedium is True:
                print('\nRT_k1_w: (AdjointMedium is True) and (normalisation=\'pressure\')')
                print(72*'-')
                print('We have defined the scattering coefficients of the adjoint medium only for flux-normalisation.\n')
            
        elif (self.ReciprocalMedium is False) and (normalisation is 'flux'):
            
            # True medium (no need to coorect (w,k1)=(0,0) element)
            denom = 1/(K3_u*beta_l + K3_l*beta_u)
            rP = (K3_u*beta_l - K3_l*beta_u) *denom
            tP = 2*(K3_u*beta_l*K3_l*beta_u)**0.5 * denom
            rM = -rP
            tM = tP
            
            if self.AdjointMedium is True:
                # Adjoint medium (no need to coorect (w,k1)=(0,0) element)
                denom = 1/(K3n_u*beta_l + K3n_l*beta_u)
                rPa = (K3n_u*beta_l - K3n_l*beta_u) * denom
                tPa = 2*(K3n_u*beta_l*K3n_l*beta_u)**0.5 * denom
                rMa = -rPa
                tMa = tPa 
            
        elif (self.ReciprocalMedium is False) and (normalisation is 'pressure'):
            
            # True medium  (no need to coorect (w,k1)=(0,0) element)
            denom = 1/(K3_u*beta_l + K3_l*beta_u)
            rP = (K3_u*beta_l - K3_l*beta_u) * denom
            tP = 2*K3_u*beta_l * denom
            rM = -rP
            tM = 2*K3_l*beta_u * denom
            
            if self.verbose is True and self.AdjointMedium is True:
                print('\nRT_k1_w: (AdjointMedium is True) and (normalisation=\'pressure\')')
                print(72*'-')
                print('We have defined the scattering coefficients of the adjoint medium only for flux-normalisation.\n')
            
        out = {'rP':rP,'tP':tP,'rM':rM,'tM':tM,'rPa':rPa,'tPa':tPa,'rMa':rMa,'tMa':tMa}
        return out
    
    def W_propagators_k1_w(self,K3=None,K3n=None,g3=None,dx3=None):
        """computes the downgoing propagator 'wP' and the upgoing progagator 'wM' for all sampled vertical-wavenumbers 'K3' and a vertical distance 'dx3' (downward pointing :math:`x_3`-axis).
        
        
        Parameters
        ----------
    
        K3 : numpy.ndarray
            Vertical-wavenumber meshgrid :math:`k_3` for all frquencies :math:`\omega` and all horizontal-wavenumbers :math:`k_1`.
        
        K3n : inumpy.ndarray, optional (required if 'AdjointMedium=True')
            Vertical-wavenumber meshgrid :math:`k_3` for all frquencies :math:`\omega` and all sign-inverted horizontal-wavenumbers :math:`k_1`.
            
        g3 : int, float
            Medium parameter :math:`\gamma_3`.
            
        dx3 : int, float
            Vertical propagation distance :math:`\Delta x_3` (downward pointing :math:`x_3`-axis).
            
        Returns
        -------
    
        dict
            Dictionary that contains 
                - **wP**: Downward propagator :math:`\\tilde{w}^+`.
                - **wM**: Upward propagator :math:`\\tilde{w}^-`.
                - **wPa**: Downward propagator :math:`\\tilde{w}^{+(a)}` (adjoint medium). 
                - **wMa**: Upward propagator :math:`\\tilde{w}^{-(a)}` (adjoint medium). 
            All propagators are stored in an arrays of shape (nf,nr). The variables 'wPa' and 'wMa' are computed for the setting 'AdjointMedium=True'.
        
        
        .. todo::
            
            In a non-reciprocal medium, for a complex-valued frequency 
            :math:`\omega'=\omega+\mathrm{j}\epsilon` one of the propagators 
            has an exponentially growing term. Does that cause errors? If yes, 
            can wefix that manually?
        
        
        References
        ----------
        Kees document as soon as it is published.


        Examples
        --------
        
        >>> from Layered_NRM_k1_w import Layered_NRM_k1_w as LM
        >>> import numpy as np
        
        >>> # Initialise a wavefield
        >>> F=LM(nt=1024,dt=0.005,nr=512,dx1=12.5,x3vec=np.array([1.1,2.2,3.7]),
        >>>      avec=np.array([1,2,3])*1e-3,bvec=np.array([1.4,3.14,2])*1e-4,
        >>>      g1vec=1j*np.array([0.8,2,1.3])*1e-4,
        >>>      g3vec=1j*np.array([1.8,0.7,2.3])*1e-4,
        >>>      ReciprocalMedium=True,AdjointMedium=True)

        >>> # Compute the propagators of the first layer
        >>> Prop = F.W_propagators_k1_w(K3=F.K3[:,:,0],K3n=F.K3n[:,:,0],
        >>>                             g3=F.g3vec[0],dx3=F.x3vec[0])

        >>> wP = Prop['wP']
        >>> wM = Prop['wM']
        
        >>> # In reciprocal media the down- and upgoing propagators are identical
        >>> np.linalg.norm(wP-wM)
        0.0
        
       
        """
        
        # Check if required input variables are given
        if (    (np.shape(K3) != (self.nf,self.nr)) 
             or (not np.isscalar(g3)) or (not np.isscalar(dx3)) ):
            sys.exit('W_propagators_k1_w: The input variables \'g3\' and  '+
                     '\'dx3\' must be scalars. The input variable \'K3\' must'+
                     ' have the shape (nf,nr)=(%d,%d).'%(self.nf,self.nr))
            
        # If AdjointMedium=True it is required to set K3n=K3(-k1)
        if (self.AdjointMedium is True) and (np.shape(K3n) != (self.nf,self.nr)):
            sys.exit('W_propagators_k1_w: If \'AdjointMedium=True\' the input'+
                     'variable \'K3n\' must be given, and it must have the '+
                      'shape (nf,nr)=(%d,%d).'%(self.nf,self.nr))
        
        if self.ReciprocalMedium is True:
            
            wP = np.exp(1j*K3*dx3)
            wM = wP.copy()
            
            if self.AdjointMedium is True:
                wPa = np.exp(1j*K3n*dx3)
                wMa = wPa.copy()
            else: 
                wPa = None
                wMa = None
        
        elif self.ReciprocalMedium is False:
            
            # Frequency meshgrid
            Om = self.W_K1_grid()['Wgrid']
            
            wP = np.exp(1j*(K3+Om*g3)*dx3)
            wM = np.exp(1j*(K3-Om*g3)*dx3)
        
            if self.AdjointMedium is True:
                wPa = np.exp(1j*(K3n-Om*g3)*dx3)
                wMa = np.exp(1j*(K3n+Om*g3)*dx3)
            else: 
                wPa = None
                wMa = None
                
        out = {'wP':wP,'wM':wM,'wPa':wPa,'wMa':wMa}
        return out
    
    def Contains_Nan_Inf(self,FuncName,*args):
        """checks if the given arrays contain NaN or Inf elements. If an array
        contains NaN or Inf elements a command line statement is printed.
        
        Parameters
        ----------
        
        FuncName : str
            Name of the function in which **Contains_Nan_Inf** is called.
            
        *args : tuple
            Undetermined number of input tuples. The first tuple element is an
            array, the second tuple element is the name of the array.
            
        Yields
        ------
        
        Message : str or None
            If any of the input arrays contains a NaN or Inf, a message is 
            printed in the command line.
            
        Notes
        -----
        
        This function is only meant for internal usage. Therefore, there are no
        checks of the input variables.
        """
        keep = 0
        
        for i in np.arange(0,len(args)):
            Var  = args[i][0]
            Name = args[i][1]
            
            if np.isnan(Var).any() or np.isinf(Var).any() :
                
                if keep == 0:
                    print('\n'+FuncName+':\n'+72*'-'+'\n')
                    print('At least one element of the modelled wavefields' 
                          + 'contains a NaN (Not a Number) or an Inf '
                          + '(infinite).\n')
                    keep += 1
                
                if np.isnan(Var).any():
                    print('\t - '+Name+' contains %d NaN.'
                          %int(np.count_nonzero(np.isnan(Var))))
                if np.isinf(Var).any():
                    print('\t - '+Name+' contains %d Inf.'
                          %int(np.count_nonzero(np.isinf(Var))))
        return
            
    
    def RT_response_k1_w(self,x3vec=None,avec=None,bvec=None,g1vec=None,g3vec=None,
                         normalisation='flux',InternalMultiples=True):
        """computes the reflection and transmission responses from above and 
        from below. The medium parameters defined in **Layered_NRM_k1_w** are 
        used, except if the medium parameters are given via the input 
        variables. 
        
        The medium responses are associated to measurements at :math:`x_3=0` 
        and at :math:`x_3=` 'x3vec[-2]' :math:`+\epsilon`, where 
        :math:`\epsilon` is an infinitesimally small positive constant. Hence, 
        the propagation from :math:`x_3=0` to the shallowest interface is 
        included. However, the propagation through the deepest layer is 
        excluded.
        
        Parameters
        ----------
    
        x3vec : numpy.ndarray, optional
            Vertical spatial vector :math:`x_3`, for n layers 'x3vec' must have 
            the shape (n,). We define the :math:`x_3`-axis as 
            downward-pointing. Implicitly, the first value on the 
            :math:`x_3`-axis is zero (not stored in 'x3vec').
    
        avec : numpy.ndarray, optional
            Medium parameter :math:`\\alpha` (real-valued), for n layers 'avec' 
            must have the shape (n,).
            
        bvec : numpy.ndarray, optional
            Medium parameter :math:`\\beta` (real-valued), for n layers 'bvec' 
            must have the shape (n,).
        
        g1vec : numpy.ndarray, optional
            Medium parameter :math:`\gamma_1` (real-valued for non-reciprocal 
            media or imaginary-valued for reciprocal media), for n layers 
            'g1vec' must have the shape (n,).
            
        g3vec : numpy.ndarray, optional
            Medium parameter :math:`\gamma_3` (real-valued for non-reciprocal 
            media or imaginary-valued for reciprocal media), for n layers 
            'g3vec' must have the shape (n,).
            
        normalisation : str, optional
            For pressure-normalisation set normalisation='pressure', for 
            flux-normalisation set normalisation='flux'.
            
        InternalMultiples : bool, optional
            To model internal multiples set 'InternalMultiples=True'. To 
            ignore internal multiples set 'InternalMultiples=False'.
            
        Returns
        -------
    
        dict
            Dictionary that contains 
                - **RP**: Reflection response from above.
                - **TP**: Transmission response from above.
                - **RM**: Reflection response from below.
                - **TM**: Transmission response from below.
                - **RPa**: Reflection response from above (adjoint medium).
                - **TPa**: Transmission response from above (adjoint medium).
                - **RMa**: Reflection response from below (adjoint medium).
                - **TMa**: Transmission response from below (adjoint medium).
            All medium responses are stored in arrays of shape (nf,nr). The 
            variables 'RPa', 'TPa', 'RMa' and 'TMa' are computed only if one 
            sets 'AdjointMedium=True'.
        
        References
        ----------
        Kees document as soon as it is published.
        
        Examples
        --------
        
        >>> from Layered_NRM_k1_w import Layered_NRM_k1_w as LM
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        
        >>> # Initialise a wavefield in a 1D reciprocal medium
        >>> F = LM(nt=4096,dt=0.005,nr=2048,dx1=12.5,eps=2/(2049*0.005),
                   x3vec=np.array([1.1,2.2,3.7])*1e3,
                   avec=np.array([1,2,3])*1e-3,bvec=np.array([1.4,3.14,2])*1e-4,
                   g1vec=np.array([0.8,2,1.3])*1e-4,
                   g3vec=np.array([1.8,0.7,2.3])*1e-4,
                   ReciprocalMedium=False,AdjointMedium=True) 
        
        >>> # Model the medium responses
        >>> RT = F.RT_response_k1_w(normalisation='flux',InternalMultiples=True)
        >>> RP = RT['RP']
        >>> # Make a Ricker wavelet
        >>> Wav=F.RickerWavelet_w(f0=30)
        >>> # Compute gain function (correct for complex-valued frequency)
        >>> gain = F.Gain_t()
        >>> # Apply wavelet, transform to the space-time domain and apply gain
        >>> rP = np.fft.fftshift(gain*F.K1W2X1T(Wav*RP),axes=(0,1))
        
        >>> # Plot reflection response
        >>> T   = F.Tvec()['tvec']
        >>> X1  = F.Xvec()['xvec']
        >>> ex  = (X1[0,0], X1[-1,0], T[-1,0], T[0,0])
        >>> asp = X1[-1,0]/T[-1,0]
        >>> plt.figure(); plt.imshow(rP,cmap='seismic',vmin=-1e-3,vmax=1e-3,
                                     extent=ex,aspect=asp)
        >>> plt.xlabel('Offset (m)')
        >>> plt.ylabel('Time (s)')
        
        .. image:: ../pictures/cropped/Rplus_k1_w.png
            :height: 200px
            :width: 200 px
        
        
        """
        
        # Check if normalisation is set correctly
        if (normalisation is not 'flux') and (normalisation is not 'pressure'):
            sys.exit('RT_response_k1_w: The input variable \'normalisation\' '+
                     'must be set, either to \'flux\', or to \'pressure\'.')
            
        # Medium responses of the adjoint medium can only be computed in 
        # flux-normalisation
        if (self.AdjointMedium is True) and (normalisation is 'pressure'):
            sys.exit('RT_response_k1_w: We have defined the scattering '+
                     'coefficients of the adjoint medium only for '     +
                     'flux-normalisation.')
        
        # Check if a layer stack is given
        if (isinstance(x3vec,np.ndarray) and isinstance(avec,np.ndarray) 
        and isinstance(bvec,np.ndarray) and isinstance(g1vec,np.ndarray) 
        and isinstance(g3vec,np.ndarray)):
            
            # Create a wavefield in a sub-medium
            # I do this because when the sub-wavefield is initialised all 
            # parameters are automatically tested for correctness
            self.SubSelf = Layered_NRM_k1_w(self.nt,self.dt,self.nr,self.dx1,
                                            self.verbose,eps=self.eps,
                                            x3vec=x3vec,avec=avec,
                                            bvec=bvec,g1vec=g1vec,g3vec=g3vec,
                                            ReciprocalMedium=self.ReciprocalMedium,
                                            AdjointMedium=self.AdjointMedium)
            
            x3vec = self.SubSelf.x3vec
            bvec  = self.SubSelf.bvec
            g3vec = self.SubSelf.g3vec
            K3    = self.SubSelf.K3
            K3n   = self.SubSelf.K3n
                
        # Else compute response of entire medium
        else:
            x3vec = self.x3vec
            bvec  = self.bvec
            g3vec = self.g3vec
            K3    = self.K3
            K3n   = self.K3n
            
        
        # Number of layers
        N = np.size(x3vec)
        
        # Vector with layer thicknesses
        dx3vec = x3vec.copy()
        dx3vec[1:] = x3vec[1:]-x3vec[:-1]
        
        # Reflection responses: Initial value
        RP = np.zeros((self.nf,self.nr),dtype=complex)
        RM = np.zeros((self.nf,self.nr),dtype=complex)
        
        # Here every frequency component has an amplitude equal to one. Hence,
        # the total wavefield has a strength of sqrt(nt*nr)
        # When an ifft is applied the wavefield is scaled by 1/sqrt(nt*nr).
        # Hence in the time domain the wavefield has an amplitude equal to one.
        TP = np.ones((self.nf,self.nr),dtype=complex)
        TM = np.ones((self.nf,self.nr),dtype=complex)
        
        # Internal multiple operator: Initial value
        M1 = np.ones((self.nf,self.nr),dtype=complex)
        M2 = np.ones((self.nf,self.nr),dtype=complex)
        
        if self.AdjointMedium is True:
            RPa = RP.copy()
            RMa = RM.copy()
            TPa = TP.copy()
            TMa = TM.copy()
        else:
            RPa = None
            TPa = None
            RMa = None
            TMa = None
        
        # Loop over N-1 interfaces
        for n in range(0,N-1):
            
            # Scattering coefficients
            ScatCoeffs = self.RT_k1_w(beta_u=bvec[n],g3_u=g3vec[n],
                                     K3_u=K3[:,:,n],K3n_u=K3n[:,:,n],
                                     beta_l=bvec[n+1],g3_l=g3vec[n+1],
                                     K3_l=K3[:,:,n+1],K3n_l=K3n[:,:,n+1],
                                     normalisation=normalisation)
            
            rP = ScatCoeffs['rP']
            tP = ScatCoeffs['tP']
            rM = ScatCoeffs['rM']
            tM = ScatCoeffs['tM']
            
            # Propagators
            W = self.W_propagators_k1_w(K3=K3[:,:,n],K3n=K3n[:,:,n],
                                        g3=g3vec[n],dx3=dx3vec[n]    )
            WP = W['wP']
            WM = W['wM']
            
            if InternalMultiples is True:
                M1 = 1 / (1 - RM*WM*rP*WP)
                M2 = 1 / (1 - rP*WP*RM*WM)
            
            # Update reflection / transmission responses
            RP = RP + TM*WM*rP*WP*M1*TP
            RM = rM + tP*WP*RM*WM*M2*tM
            TP = tP*WP*M1*TP
            TM = TM*WM*M2*tM  
            
            if self.AdjointMedium is True:
                rP = ScatCoeffs['rPa']
                tP = ScatCoeffs['tPa']
                rM = ScatCoeffs['rMa']
                tM = ScatCoeffs['tMa']
                WP = W['wPa']
                WM = W['wMa']
            
                if InternalMultiples is True:
                    M1 = 1 / (1 - RMa*WM*rP*WP)
                    M2 = 1 / (1 - rP*WP*RMa*WM)
                
                # Update reflection / transmission responses
                RPa = RPa + TMa*WM*rP*WP*M1*TPa
                RMa = rM + tP*WP*RMa*WM*M2*tM
                TPa = tP*WP*M1*TPa
                TMa = TMa*WM*M2*tM  
                
        # Verbose: Inform the user if any wavefield contains NaNs of Infs.
        if self.verbose is True:
            self.Contains_Nan_Inf('RT_response_k1_w',(RP,'RP'),(TP,'TP'),
                                  (RM,'RM'),(TM,'TM'))
            
            if self.AdjointMedium is True:
                self.Contains_Nan_Inf('RT_response_k1_w',(RPa,'RPa'),
                                      (TPa,'TPa'),(RMa,'RMa'),(TMa,'TMa'))

                
        out={'RP':RP  ,'TP':TP  ,'RM':RM  ,'TM':TM,
             'RPa':RPa,'TPa':TPa,'RMa':RMa,'TMa':TMa}
        return out
            
    # Insert a layer in the model    
    def Insert_layer(self,x3,UpdateSelf=False):
        """inserts a transparent interface at the depth level 'x3'. If 'x3' 
        coincides with an interface of the model, the model's interface is left 
        unchanged. If 'x3' is a vector it is interpreted as multiple depth 
        levels, at each one a transparent interface will be inserted.
        
        Parameters
        ----------
    
        x3 : int, float, numpy.ndarray
            A depth level, or a vector of depth levels, at which a transparent 
            interface will be inserted. The variable 'x3' either must be a 
            scalar, or have the shape (n,). Each element of 'x3' must be 
            real-valued and greater than, or equal to zero.
    
        UpdateSelf : bool, optional
            Set 'UpdateSelf=True' to not only output an updated model but also 
            update the 'self' parameters.
            
        Returns
        -------
    
        dict
            Dictionary that contains 
                - **x3vec**: Updated depth vector.
                - **avec**:  Updated :math:`\\alpha` vector.
                - **bvec**:  Updated :math:`\\beta` vector.
                - **g1vec**: Updated :math:`\gamma_1` vector.
                - **g3vec**: Updated :math:`\gamma_3` vector.
                - **K3**:    Updated :math:`k_3(k_1)` vector.
                - **K3n**:   Updated :math:`k_3(-k_1)` vector.
            All medium parameter vectors are stored in arrays of shape (n,).
            The vertical wavenumbers are stored in arrays of shape (nf,nr,n).
        
        Examples
        --------

        >>> from Layered_NRM_k1_w import Layered_NRM_k1_w as LM
        >>> import numpy as np
        
        >>> # Initialise a wavefield in a 1D reciprocal medium
        >>> F=LM(nt=1024,dt=0.005,nr=512,dx1=10,
                 x3vec=np.array([10,150,200]),
                 avec=np.array([1,2,3]),bvec=np.array([0.4,3.14,2]),
                 g1vec=np.array([0.9,2.1,0.3]),g3vec=np.array([0.7,1.14,0.2]))
        
        >>> # Insert a transparent layer at x3=1
        >>> out=F.Insert_layer(x3=1,UpdateSelf=False)
        
        >>> # Updated depth vector
        >>> out['x3vec']
        array([  1,  10, 150, 200])
        
        >>> # Updated alpha vector
        >>> out['avec']
        array([1, 1, 2, 3])
        
        """
        
        # Check if x3 is a scalar or an array of the shape (n,).
        if not (    isinstance(x3,int) 
                or  isinstance(x3,float) 
                or (isinstance(x3,np.ndarray) and x3.ndim == 1) ):
            sys.exit('Insert_layer: The input variable \'x3\' must be either '
                     +'a scalar, or an array of shape (n,).')
        
        if isinstance(x3,int) or  isinstance(x3,float):
            x3 = np.array([x3])
        
        # Check if x3 is real-valued.
        if not np.isreal(x3).all():
            sys.exit('Insert_layer: The input variable \'x3\' must be '
                     +'real-valued.')
        
        # Check if all elements of x3 are greater than, or equal to zero.
        if x3[x3<0].size > 0:
            sys.exit('Insert_layer: Each element of the input variable \'x3\' '
                     +'must be  greater than, or equal to zero.')
        
        X3vec = self.x3vec
        Avec  = self.avec
        Bvec  = self.bvec
        G1vec = self.g1vec
        G3vec = self.g3vec
        K3    = self.K3
        K3n   = self.K3n
        
        for i in range(np.size(x3)):
        
            # Vector of depths smaller than or equal to x3[i]
            L = X3vec[X3vec<=x3[i]] 
            
            # Case1: x3[i] smaller than X3vec[0]
            if L.size == 0:
                X3vec = np.hstack([x3[i]   ,X3vec])
                Avec  = np.hstack([Avec[0] ,Avec])
                Bvec  = np.hstack([Bvec[0] ,Bvec])
                G1vec = np.hstack([G1vec[0],G1vec])
                G3vec = np.hstack([G3vec[0],G3vec])
                K3    = np.dstack([K3[:,:,:1]   ,K3])
                K3n   = np.dstack([K3n[:,:,:1]  ,K3n])
            
            # Case2: x3[i] coincides with an element of X3vec
            elif L[-1] == x3[i]:
                X3vec = X3vec
                Avec  = Avec
                Bvec  = Bvec
                G1vec = G1vec
                G3vec = G3vec
                K3    = K3
                K3n   = K3n
            
            # Case 3: x3[i] is larger than X3vec[-1]
            elif L.size == X3vec.size:
                X3vec = np.hstack([X3vec,x3[i]])
                Avec  = np.hstack([Avec ,Avec[-1]])
                Bvec  = np.hstack([Bvec ,Bvec[-1]])
                G1vec = np.hstack([G1vec,G1vec[-1]])
                G3vec = np.hstack([G3vec,G3vec[-1]])
                K3    = np.dstack([K3   ,K3[:,:,-1:]])
                K3n   = np.dstack([K3n  ,K3n[:,:,-1:]])
                
            # Case 4: x3[i] is between X3vec[0] and X3vec[-1] AND does not 
            # coincide with any element of X3vec
            else:
                
                b = L[-1] 
                ind = X3vec.tolist().index(b)
                
                X3vec = np.hstack([X3vec[:ind+1],x3[i]       ,X3vec[ind+1:]])
                Avec  = np.hstack([Avec[:ind+1] ,Avec[ind+1] ,Avec[ind+1:]])
                Bvec  = np.hstack([Bvec[:ind+1] ,Bvec[ind+1] ,Bvec[ind+1:]])
                G1vec = np.hstack([G1vec[:ind+1],G1vec[ind+1],G1vec[ind+1:]])
                G3vec = np.hstack([G3vec[:ind+1],G3vec[ind+1],G3vec[ind+1:]])
                K3    = np.dstack([K3[:,:,:ind+1]   ,K3[:,:,ind+1:ind+2]   ,K3[:,:,ind+1:]])
                K3n   = np.dstack([K3n[:,:,:ind+1]  ,K3n[:,:,ind+1:ind+2]  ,K3n[:,:,ind+1:]])
            
        # Update self: Apply layer insertion to the self-parameters    
        if UpdateSelf is True:
            self.x3vec = X3vec
            self.avec  = Avec
            self.bvec  = Bvec
            self.g1vec = G1vec
            self.g3vec = G3vec
            self.K3    = K3
            self.K3n   = K3n
            
        out = {'x3vec':X3vec,'avec':Avec,'bvec':Bvec,
               'g1vec':G1vec,'g3vec':G3vec,'K3':K3,'K3n':K3n}
        return out
    
    def GreensFunction_k1_w(self,x3R,x3S,normalisation='flux',
                           InternalMultiples=True):
        """computes the one-way Green\'s functions for a receiver and source 
        depth defined by the input variables \'x3R\' and \'x3S\'. The one-way 
        wavefields are decomposed at the receiver- and at the source-side. We 
        define the receiver and source depths just below \'x3R\' and \'x3S\', 
        respectively (this is important if the receiver or source depth 
        coincides with an interface).
        
        Parameters
        ----------
    
        x3R : int,float
            Receiver depth.
    
        x3S : int, float
            Source depth.
            
        normalisation : str, optional
            For pressure-normalisation set normalisation='pressure', for 
            flux-normalisation set normalisation='flux'.
            
        InternalMultiples : bool, optional
            To model internal multiples set 'InternalMultiples=True'. To 
            exclude internal multiples set 'InternalMultiples=False'.
            
        Returns
        -------
    
        dict
            Dictionary that contains 
                - **GPP**: Green\'s function :math:`G^{+,+}` (true medium).
                - **GPM**: Green\'s function :math:`G^{+,-}` (true medium).
                - **GMP**: Green\'s function :math:`G^{-,+}` (true medium).
                - **GMM**: Green\'s function :math:`G^{-,-}` (true medium).
                - **GPPa**: Green\'s function :math:`G^{+,+}` (adjoint medium).
                - **GPMa**: Green\'s function :math:`G^{+,-}` (adjoint medium).
                - **GMPa**: Green\'s function :math:`G^{-,+}` (adjoint medium).
                - **GMMa**: Green\'s function :math:`G^{-,-}` (adjoint medium).
            All medium responses are stored in arrays of shape (nf,nr). The 
            variables 'GPPa', 'GPMa', 'GMPa' and 'GMMa' are computed, only if 
            one sets 'AdjointMedium=True'.
        
        Notes
        -----
        
        - The superscript \'+\' and \'-\' refer to downgoing and upgoing waves, 
        respectively.
        - The first superscript refers to the wavefield at the receiver-side.
        - The second superscript refers to the wavefield at the source-side.
        
        References
        ----------
        Kees document as soon as it is published.
        
        Examples
        --------

        >>> to be done
        >>> from Layered_NRM_k1_w import Layered_NRM_k1_w as LM
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        
        >>> # Initialise a wavefield in a 1D reciprocal medium
        >>> F = LM(nt=4096,dt=0.005,nr=2048,dx1=12.5,eps=2/(2049*0.005),
                   x3vec=np.array([1.1,2.2,3.7])*1e3,
                   avec=np.array([1,2,3])*1e-3,bvec=np.array([1.4,3.14,2])*1e-4,
                   g1vec=np.array([0.8,2,1.3])*1e-4,
                   g3vec=np.array([1.8,0.7,2.3])*1e-4,
                   ReciprocalMedium=False,AdjointMedium=True) 
        
        >>> G = F.GreensFunction_p_w(x3R=0,x3S=0,normalisation=normalisation,
        >>>                          InternalMultiples=InternalMultiples)
        >>> RT=F.RT_response_p_w(normalisation=normalisation,
        >>>                      InternalMultiples=InternalMultiples)
        >>> np.linalg.norm(RT['RP']-G['GMP']
        0.0
        
        """
        
        # Insert transparent interfaces at source and receiver depth levels
        # The insertion implicitly checks that x3R and x3S are non-negative 
        # real-valued scalars
        # If the receiver or source depth is greater than, or equal to the 
        # deepest interface, we insert another transparent layer below the  
        # 'new' deepest interface. This is necessary because the function 
        # RT_response_k1_w does not compute the propagation through the deepest
        # layer. By adding a transparent interface below the source/receiver we
        # ensure that the propagation is computed correctly.
        if (x3R >= self.x3vec[-1]) or (x3S >= self.x3vec[-1]):
            xb = np.max([x3R,x3S])+1
            Tmp_medium = self.Insert_layer(x3=np.array([x3R,x3S,xb]),
                                           UpdateSelf=False)
        else:
            Tmp_medium = self.Insert_layer(x3=np.array([x3R,x3S]),
                                           UpdateSelf=False)
            
        X3vec = Tmp_medium['x3vec']
        Avec  = Tmp_medium['avec']
        Bvec  = Tmp_medium['bvec']
        G1vec = Tmp_medium['g1vec']
        G3vec = Tmp_medium['g3vec']
        
        # Get indices of the receiver and source interfaces
        r = X3vec.tolist().index(x3R)
        s = X3vec.tolist().index(x3S)
        
        if x3R > x3S:
            
            # Overburden
            x3vec = X3vec[:s+2]
            avec  = Avec[:s+2]
            bvec  = Bvec[:s+2]
            g1vec = G1vec[:s+2]
            g3vec = G3vec[:s+2]
            
            L1 = self.RT_response_k1_w(x3vec=x3vec,avec=avec,bvec=bvec,
                                      g1vec=g1vec,g3vec=g3vec,
                                      normalisation=normalisation,
                                      InternalMultiples=InternalMultiples)

            # Sandwiched layer stack
            x3vec = X3vec[s+1:r+2] - X3vec[s]
            avec  = Avec[s+1:r+2]
            bvec  = Bvec[s+1:r+2]
            g1vec = G1vec[s+1:r+2]
            g3vec = G3vec[s+1:r+2]
            
            L2 = self.RT_response_k1_w(x3vec=x3vec,avec=avec,bvec=bvec,
                                      g1vec=g1vec,g3vec=g3vec,
                                      normalisation=normalisation,
                                      InternalMultiples=InternalMultiples)
            
            # Underburden
            x3vec = X3vec[r+1:] - X3vec[r]
            avec  = Avec[r+1:]
            bvec  = Bvec[r+1:]
            g1vec = G1vec[r+1:]
            g3vec = G3vec[r+1:]
            
            # Exception if underburden is homogeneous
            if x3vec.size == 0:
                x3vec = np.array([0])
                avec  = Avec[r:]
                bvec  = Bvec[r:]
                g1vec = G1vec[r:]
                g3vec = G3vec[r:]
            
            L3 = self.RT_response_k1_w(x3vec=x3vec,avec=avec,bvec=bvec,
                                      g1vec=g1vec,g3vec=g3vec,
                                      normalisation=normalisation,
                                      InternalMultiples=InternalMultiples)

            # Get variables that are used multiple times to avoid multiple 
            # reading from dictionary
            RM1 = L1['RM']
            TP2 = L2['TP']
            
            if InternalMultiples is True:
                M1 = 1 / (1 - RM1*L2['RP'])
            else:
                M1 = 1
            GPP12 =  TP2*M1
            GPM12 = -TP2*M1*RM1 # Multiply by -1 because upgoing
                                # sources are defined with 
                                # negative amplitude
                                            
            # Compute reflection from below of parts 1+2 
            RM12 = L2['RM'] + TP2*M1*RM1*L2['TM']
            
            # Compute the Green's functions G13 for the complete medium
            if InternalMultiples is True:
                M2 = 1 / ( 1 - RM12*L3['RP'] )
            else:
                M2 = 1
            GPP13 = M2*GPP12
            GPM13 = M2*GPM12
            GMP13 = L3['RP']*M2*GPP12
            GMM13 = L3['RP']*M2*GPM12
            
            # Green;s functions in adjoint medium
            if self.AdjointMedium is True:
                # Get variables that are used multiple times to avoid multiple 
                # reading from dictionary
                RM1 = L1['RMa']
                TP2 = L2['TPa']
                
                if InternalMultiples is True:
                    M1 = 1 / (1 - RM1*L2['RPa'])
                else:
                    M1 = 1
                GPP12 =  TP2*M1
                GPM12 = -TP2*M1*RM1 # Multiply by -1 because upgoing
                                    # sources are defined with 
                                    # negative amplitude
                                                
                # Compute reflection from below of parts 1+2 
                RM12 = L2['RMa'] + TP2*M1*RM1*L2['TMa']
                
                # Compute the Green's functions G13 for the complete medium
                if InternalMultiples is True:
                    M2 = 1 / ( 1 - RM12*L3['RPa'] )
                else:
                    M2 = 1
                GPP13a = M2*GPP12
                GPM13a = M2*GPM12
                GMP13a = L3['RPa']*M2*GPP12
                GMM13a = L3['RPa']*M2*GPM12
                
            
            
        elif x3R == x3S:
            
            # Overburden
            x3vec = X3vec[:s+2]
            avec  = Avec[:s+2]
            bvec  = Bvec[:s+2]
            g1vec = G1vec[:s+2]
            g3vec = G3vec[:s+2]
            
            L1 = self.RT_response_k1_w(x3vec=x3vec,avec=avec,bvec=bvec,
                                      g1vec=g1vec,g3vec=g3vec,
                                      normalisation=normalisation,
                                      InternalMultiples=InternalMultiples)
            
            # Underburden
            x3vec = X3vec[r+1:] - X3vec[r]
            avec  = Avec[r+1:]
            bvec  = Bvec[r+1:]
            g1vec = G1vec[r+1:]
            g3vec = G3vec[r+1:]
            
            # Exception if underburden is homogeneous
            if x3vec.size == 0:
                x3vec = np.array([0])
                avec  = Avec[r:]
                bvec  = Bvec[r:]
                g1vec = G1vec[r:]
                g3vec = G3vec[r:]
            
            L3 = self.RT_response_k1_w(x3vec=x3vec,avec=avec,bvec=bvec,
                                      g1vec=g1vec,g3vec=g3vec,
                                      normalisation=normalisation,
                                      InternalMultiples=InternalMultiples)
          
            # Get variables that are used multiple times to avoid multiple 
            # reading from dictionary
            RM1 = L1['RM']
            RP3 = L3['RP']
            
            if InternalMultiples is True:
                M1 = 1 / (1 - RM1*RP3 )
                M2 = 1 / (1 - RP3*RM1 )
            else:
                M1 = np.ones((self.nf,self.nr),dtype=complex)
                M2 = M1.copy()
                
            GPP13 =  M1.copy() 
            GPM13 = -M1*RM1     # Multiply by -1 because upgoing
                                # sources are defined with 
                                # negative amplitude
            GMP13 =  RP3*M1
            GMM13 = -M2     # Multiply by -1 because upgoing
                            # sources are defined with 
                            # negative amplitude
                            
            # Green;s functions in adjoint medium
            if self.AdjointMedium is True:
                # Get variables that are used multiple times to avoid multiple 
                # reading from dictionary
                RM1 = L1['RMa']
                RP3 = L3['RPa']
                
                if InternalMultiples is True:
                    M1 = 1 / (1 - RM1*RP3 )
                    M2 = 1 / (1 - RP3*RM1 )
                else:
                    M1 = np.ones((self.nf,self.nr),dtype=complex)
                    M2 = M1.copy()
                    
                GPP13a =  M1.copy() 
                GPM13a = -M1*RM1     # Multiply by -1 because upgoing
                                     # sources are defined with 
                                     # negative amplitude
                GMP13a =  RP3*M1
                GMM13a = -M2     # Multiply by -1 because upgoing
                                 # sources are defined with 
                                 # negative amplitude
            
        elif x3R < x3S:
            
            # Overburden
            x3vec = X3vec[:r+2]
            avec  = Avec[:r+2]
            bvec  = Bvec[:r+2]
            g1vec = G1vec[:r+2]
            g3vec = G3vec[:r+2]
            
            L1 = self.RT_response_k1_w(x3vec=x3vec,avec=avec,bvec=bvec,
                                      g1vec=g1vec,g3vec=g3vec,
                                      normalisation=normalisation,
                                      InternalMultiples=InternalMultiples)
            
            # Sandwiched layer stack
            x3vec = X3vec[r+1:s+2] - X3vec[r]
            avec  = Avec[r+1:s+2]
            bvec  = Bvec[r+1:s+2]
            g1vec = G1vec[r+1:s+2]
            g3vec = G3vec[r+1:s+2]
            
            L2 = self.RT_response_k1_w(x3vec=x3vec,avec=avec,bvec=bvec,
                                      g1vec=g1vec,g3vec=g3vec,
                                      normalisation=normalisation,
                                      InternalMultiples=InternalMultiples)
            
            # Underburden
            x3vec = X3vec[s+1:] - X3vec[s]
            avec  = Avec[s+1:]
            bvec  = Bvec[s+1:]
            g1vec = G1vec[s+1:]
            g3vec = G3vec[s+1:]
            
            # Exception if underburden is homogeneous
            if x3vec.size == 0:
                x3vec = np.array([0])
                avec  = Avec[s:]
                bvec  = Bvec[s:]
                g1vec = G1vec[s:]
                g3vec = G3vec[s:]
            
            L3 = self.RT_response_k1_w(x3vec=x3vec,avec=avec,bvec=bvec,
                                      g1vec=g1vec,g3vec=g3vec,
                                      normalisation=normalisation,
                                      InternalMultiples=InternalMultiples)
            
            # Get variables that are used multiple times to avoid multiple 
            # reading from dictionary
            RM1 = L1['RM']
            TM2 = L2['TM']
            RP3 = L3['RP']
            
            # Compute reflection from above of part 2+3 
            if InternalMultiples is True:
                M1 = 1 / (1 - RP3*L2['RM'])
            else:
                M1 = 1
            RP23 = L2['RP'] + TM2*M1*RP3*L2['TP']
            
            # Compute the Green's functions G23 that exclude the medium above 
            # the receiver: GMP23,GMM23
            GMP23 =  TM2*M1*RP3                   
            GMM23 = -TM2*M1         # Multiply by -1 because upgoing
                                    # sources are defined with 
                                    # negative amplitude
                
            
            
            # Compute the Green's functions G13 for the complete medium
            if InternalMultiples is True:
                M2 = 1 / (1 - RP23*RM1)
            else:
                M2 = 1
            GPP13 = RM1*M2*GMP23
            GPM13 = RM1*M2*GMM23
            GMP13 = M2*GMP23
            GMM13 = M2*GMM23
            
            # Green;s functions in adjoint medium
            if self.AdjointMedium is True:
                # Get variables that are used multiple times to avoid multiple 
                # reading from dictionary
                RM1 = L1['RMa']
                TM2 = L2['TMa']
                RP3 = L3['RPa']
                
                # Compute reflection from above of part 2+3 
                if InternalMultiples is True:
                    M1 = 1 / (1 - RP3*L2['RMa'])
                else:
                    M1 = 1
                RP23 = L2['RPa'] + TM2*M1*RP3*L2['TPa']
                
                # Compute the Green's functions G23 that exclude the medium above 
                # the receiver: GMP23,GMM23
                GMP23 =  TM2*M1*RP3                   
                GMM23 = -TM2*M1         # Multiply by -1 because upgoing
                                        # sources are defined with 
                                        # negative amplitude
                    
                
                
                # Compute the Green's functions G13 for the complete medium
                if InternalMultiples is True:
                    M2 = 1 / (1 - RP23*RM1)
                else:
                    M2 = 1
                GPP13a = RM1*M2*GMP23
                GPM13a = RM1*M2*GMM23
                GMP13a = M2*GMP23
                GMM13a = M2*GMM23
                
        if self.AdjointMedium is False:
            GPP13a = None
            GPM13a = None
            GMP13a = None
            GMM13a = None
            
        # Verbose: Inform the user if any wavefield contains NaNs of Infs.
        if self.verbose is True:
            self.Contains_Nan_Inf('GreensFunction_k1_w',
                                  (GPP13,'GPP13'),(GPM13,'GPM13'),
                                  (GMP13,'GMP13'),(GMM13,'GMM13'))
            
            if self.AdjointMedium is True:
                self.Contains_Nan_Inf('GreensFunction_k1_w',
                                  (GPP13a,'GPP13a'),(GPM13a,'GPM13a'),
                                  (GMP13a,'GMP13a'),(GMM13a,'GMM13a'))
                
        out = {'GPP':GPP13  ,'GPM':GPM13  ,'GMP':GMP13  ,'GMM':GMM13,
               'GPPa':GPP13a,'GPMa':GPM13a,'GMPa':GMP13a,'GMMa':GMM13a}
        return out
    
    def FocusingFunction_p_w(self,x3F,normalisation='flux',InternalMultiples=True):
        """computes the focusing functions between the top surface (:math:`x_3=0`) and the focusing depth defined by the input variable \'x3F\'. We define the focusing depth just below \'x3F\'. Hence, if the focusing depth coincides with an interface the focusing function focuses below that interface.
        
        Parameters
        ----------
    
        x3F : int,float
            Focusing depth.
            
        normalisation : str, optional
            For pressure-normalisation set normalisation='pressure', for flux-normalisation set normalisation='flux'. Until now, this function only models the focusing function for flux-normalisation.
            
        InternalMultiples : bool, optional
            To model internal multiples set 'InternalMultiples=True'. To ignore internal multiples set 'InternalMultiples=False'.
            
        Returns
        -------
    
        dict
            Dictionary that contains 
                - **FP**: Downgoing focusing function.
                - **RP**: Reflection response from above.
                - **TP**: Transmission response from above.
                - **FM**: Upgoing focusing function.
                - **RM**: Reflection response from below.
                - **TM**: Transmission response from below.
                - **FPa**: Downgoing focusing function (adjoint medium).
                - **RPa**: Reflection response from above (adjoint medium).
                - **TPa**: Transmission response from above (adjoint medium).
                - **FMa**: Upgoing focusing function (adjoint medium).
                - **RMa**: Reflection response from below (adjoint medium).
                - **TMa**: Transmission response from below (adjoint medium).
            All medium responses are stored in arrays of shape (nf,1). The variables 'FPa', 'RPa', 'TPa', 'FMa', 'RMa' and 'TMa' are computed only if one sets 'AdjointMedium=True'.
        
        Notes
        -----
        
        - The downgoing focusing funtion :math:`\\tilde{F}_1^+` is computed by inverting the expressions for the transmission from above :math:`\\tilde{T}^+`:
            :math:`\\tilde{F}_{1,n}^+ = \\tilde{F}_{1,n-1}^+ (\\tilde{w}_n^+)^{-1} (1 - \\tilde{w}_n^+ \\tilde{R}_{n-1}^{\cap} \\tilde{w}_n^- \\tilde{r}_n^{\cup} )^{-1} (\\tilde{t}_n^+)^{-1}`
        - The upgoing focusing function is computed by applying the reflection response :math:`R^{\cup}` on the downgoing focusing funtion :math:`\\tilde{F}_1^+`:
            :math:`\\tilde{F}_{1,n}^- = \\tilde{R}^{\cup} \\tilde{F}_{1,n}^+`.
        
        References
        ----------
        Kees document as soon as it is published.
        
        Examples
        --------

        >>> from Layered_NRM_p_w import Layered_NRM_p_w as LM
        >>> import numpy as np

        >>> F=LM( nt=1024,dt=0.005,x3vec=np.array([10,150,200]),
        >>>       avec=np.array([1,2,3]),bvec=np.array([0.4,3.14,2]),
        >>>       g1vec=np.array([0.9,2.1,0.3]),g3vec=np.array([0.7,1.14,0.2]),
        >>>       p1=2e-4,ReciprocalMedium=False,AdjointMedium=True )
        
        """
        
        # Check if normalisation is set correctly
        if normalisation is not 'flux':
            sys.exit('FocusingFunction_p_w: This function only models the focusing function for flux-normalisation. (For pressure-normalistiont the required equations have to be derived.)')
        
        # Insert transparent interfaces at the focusing depth level.
        # The insertion implicitly checks that x3F is non-negative 
        # and real-valued.
        # If the focusing depth is greater than, or equal to the deepest 
        # interface, we insert another transparent layer below the focusing 
        # depth to be able to compute scattering coefficients at the focusing 
        # depth without getting an index error.
        if x3F >= self.x3vec[-1]:
            Tmp_medium = self.Insert_layer(x3=np.array([x3F,x3F+1]),
                                           UpdateSelf=False)
        else:
            Tmp_medium = self.Insert_layer(x3=x3F,UpdateSelf=False)
        
        X3vec = Tmp_medium['x3vec']
        Bvec = Tmp_medium['bvec']
        G3vec = Tmp_medium['g3vec']
        P3    = Tmp_medium['p3']
        P3n   = Tmp_medium['p3n']
        
        # Index of the focusing depth
        f = X3vec.tolist().index(x3F)
        
        # Only allow propagating waves to model the focusing function
        if np.iscomplex(P3[:f+1]).any() or np.iscomplex(P3n[:f+1]).any():
            sys.exit('FocusingFunction_p_w: (Here,) We only define the focusing function for propagating waves, i.e. not evanescent waves.')
        
        # Vector with layer thicknesses
        dx3vec = X3vec.copy()
        dx3vec[1:] = X3vec[1:]-X3vec[:-1]
        
        # Down- and upgoing focusing functions: Initial value
        # Here every frequency component has an amplitude equal to one. Hence,
        # the total wavefield has a strength of sqrt(nt)
        # When an ifft is applied the wavefield is scaled by 1/sqrt(nt).
        # Hence in the time domain the wavefield has an amplitude equal to one.
        FP = np.ones((self.nf,1),dtype=complex)
        FM = np.zeros((self.nf,1),dtype=complex)
        
        # Reflection responses: Initial value
        RP = np.zeros((self.nf,1),dtype=complex)
        RM = np.zeros((self.nf,1),dtype=complex)
        
        # Here every frequency component has an amplitude equal to one. Hence,
        # the total wavefield has a strength of sqrt(nt)
        # When an ifft is applied the wavefield is scaled by 1/sqrt(nt).
        # Hence in the time domain the wavefield has an amplitude equal to one.
        TP = np.ones((self.nf,1),dtype=complex)
        TM = np.ones((self.nf,1),dtype=complex)
        
        # Internal multiple operator: Initial value
        M1 = np.ones((self.nf,1),dtype=complex)
        M2 = np.ones((self.nf,1),dtype=complex)
        
        if self.AdjointMedium is True:
            FPa = FP.copy()
            FMa = FM.copy()
            RPa = RP.copy()
            RMa = RP.copy()
            TPa = TP.copy()
            TMa = TP.copy()
        else:
            FPa = None
            FMa = None
            RPa = None
            TPa = None
            RMa = None
            TMa = None
    
        # Loop over f+1 interfaces
        # Thus, the wavefield propagates to the focusing depth, and scatters
        # at the focusing depth.
        for n in range(0,f+1):
            
            # Scattering coefficients
            ScatCoeffs = self.RT_p_w(beta_u=Bvec[n]  ,g3_u=G3vec[n]  ,p3_u=P3[n]  ,p3n_u=P3n[n],
                                     beta_l=Bvec[n+1],g3_l=G3vec[n+1],p3_l=P3[n+1],p3n_l=P3n[n+1],
                                     normalisation=normalisation)
            
            rP = ScatCoeffs['rP']
            tP = ScatCoeffs['tP']
            rM = ScatCoeffs['rM']
            tM = ScatCoeffs['tM']
            
            # Propagators
            W = self.W_propagators_p_w(p3=P3[n],p3n=P3n[n],g3=G3vec[n],dx3=dx3vec[n])
            WP = W['wP']
            WM = W['wM']
        
            if InternalMultiples is True:
                M1 = 1 / (1 - RM*WM*rP*WP)
                M2 = 1 / (1 - rP*WP*RM*WM)
            
            # Update focusing functions and reflection / transmission responses
            FP = FP/WP*(1-WP*RM*WM*rP)/tP
            RP = RP + TM*WM*rP*WP*M1*TP
            RM = rM + tP*WP*RM*WM*M2*tM
            TP = tP*WP*M1*TP
            TM = TM*WM*M2*tM  
            FM = RP*FP
            
            # Model focusing functions in the adjoint medium
            if self.AdjointMedium is True:
                rP = ScatCoeffs['rPa']
                tP = ScatCoeffs['tPa']
                rM = ScatCoeffs['rMa']
                tM = ScatCoeffs['tMa']
                WP = W['wPa']
                WM = W['wMa']
            
                if InternalMultiples is True:
                    M1 = 1 / (1 - RMa*WM*rP*WP)
                    M2 = 1 / (1 - rP*WP*RMa*WM)
                
                # Update focusing functions and reflection / transmission responses
                FPa = FPa/WP*(1-WP*RMa*WM*rP)/tP
                RPa = RPa + TMa*WM*rP*WP*M1*TPa
                RMa = rM + tP*WP*RMa*WM*M2*tM
                TPa = tP*WP*M1*TPa
                TMa = TMa*WM*M2*tM  
                FMa = RPa*FPa
                
        # Verbose: Inform the user if any wavefield contains NaNs of Infs.
        if self.verbose is True:
            
            if (   np.isnan(FP).any() or np.isnan(FM).any() 
                or np.isnan(RP).any() or np.isnan(RM).any() 
                or np.isnan(TP).any() or np.isnan(TM).any()
                or np.isinf(FP).any() or np.isinf(FM).any()
                or np.isinf(RP).any() or np.isinf(TP).any() 
                or np.isinf(RM).any() or np.isinf(TM).any()):
                print('\n')
                print('FocusingFunction_p_w:')
                print('\n'+100*'-'+'\n')
                print('One of the modelled wavefields in the true medium contains a NaN (Not a Number) or an Inf (infinite) element.')
                print('\n')
                
                if np.isnan(FP).any():
                    print('\t - FP contains '+np.count_nonzero(np.isnan(FP))+' NaN.')
                if np.isinf(FP).any():
                    print('\t - FP contains '+np.count_nonzero(np.isinf(FP))+' Inf.')
                if np.isnan(RP).any():
                    print('\t - RP contains '+np.count_nonzero(np.isnan(RP))+' NaN.')
                if np.isinf(RP).any():
                    print('\t - RP contains '+np.count_nonzero(np.isinf(RP))+' Inf.')
                if np.isnan(TP).any():
                    print('\t - TP contains '+np.count_nonzero(np.isnan(TP))+' NaN.')
                if np.isinf(TP).any():
                    print('\t - TP contains '+np.count_nonzero(np.isinf(TP))+' Inf.')
                if np.isnan(FM).any():
                    print('\t - FM contains '+np.count_nonzero(np.isnan(FM))+' NaN.')
                if np.isinf(FM).any():
                    print('\t - FM contains '+np.count_nonzero(np.isinf(FM))+' Inf.')
                if np.isnan(RM).any():
                    print('\t - RM contains '+np.count_nonzero(np.isnan(RM))+' NaN.')
                if np.isinf(RM).any():
                    print('\t - RM contains '+np.count_nonzero(np.isinf(RM))+' Inf.')
                if np.isnan(TM).any():
                    print('\t - TM contains '+np.count_nonzero(np.isnan(TM))+' NaN.')
                if np.isinf(TM).any():
                    print('\t - TM contains '+np.count_nonzero(np.isinf(TM))+' Inf.')
            
            if self.AdjointMedium is True:
                
                if (   np.isnan(FPa).any() or np.isnan(FMa).any() 
                    or np.isnan(RPa).any() or np.isnan(RMa).any() 
                    or np.isnan(TPa).any() or np.isnan(TMa).any()
                    or np.isinf(FPa).any() or np.isinf(FMa).any()
                    or np.isinf(RPa).any() or np.isinf(TPa).any() 
                    or np.isinf(RMa).any() or np.isinf(TMa).any()):
                    print('\n')
                    print('FocusingFunction_p_w:')
                    print('\n'+100*'-'+'\n')
                    print('One of the modelled wavefields in the adoint medium contains a NaN (Not a Number) or an Inf (infinite) element.')
                    print('\n')
                    
                    if np.isnan(FPa).any():
                        print('\t - FPa contains '+np.count_nonzero(np.isnan(FPa))+' NaN.')
                    if np.isinf(FPa).any():
                        print('\t - FPa contains '+np.count_nonzero(np.isinf(FPa))+' Inf.')
                    if np.isnan(RPa).any():
                        print('\t - RPa contains '+np.count_nonzero(np.isnan(RPa))+' NaN.')
                    if np.isinf(RPa).any():
                        print('\t - RPa contains '+np.count_nonzero(np.isinf(RPa))+' Inf.')
                    if np.isnan(TPa).any():
                        print('\t - TPa contains '+np.count_nonzero(np.isnan(TPa))+' NaN.')
                    if np.isinf(TPa).any():
                        print('\t - TPa contains '+np.count_nonzero(np.isinf(TPa))+' Inf.')
                    if np.isnan(FMa).any():
                        print('\t - FMa contains '+np.count_nonzero(np.isnan(FMa))+' NaN.')
                    if np.isinf(FMa).any():
                        print('\t - FMa contains '+np.count_nonzero(np.isinf(FMa))+' Inf.')
                    if np.isnan(RMa).any():
                        print('\t - RMa contains '+np.count_nonzero(np.isnan(RMa))+' NaN.')
                    if np.isinf(RMa).any():
                        print('\t - RMa contains '+np.count_nonzero(np.isinf(RMa))+' Inf.')
                    if np.isnan(TMa).any():
                        print('\t - TMa contains '+np.count_nonzero(np.isnan(TMa))+' NaN.')
                    if np.isinf(TMa).any():
                        print('\t - TMa contains '+np.count_nonzero(np.isinf(TMa))+' Inf.')
                
                print('\n')

                
        out={'FP':FP  ,'RP':RP  ,'TP':TP  ,'FM':FM  ,'RM':RM  ,'TM':TM,
             'FPa':FPa,'RPa':RPa,'TPa':TPa,'FMa':FMa,'RMa':RMa,'TMa':TMa}
        return out
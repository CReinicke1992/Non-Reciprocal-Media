#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routines for modelling wavefields in 1.5D non-reciprocal media.

.. module:: Layered_NRM_k1_w 2\.0

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
    
                :math:`k_3=\sqrt{(\\alpha \\beta_{11} -\gamma_1^2)\omega' 
                + 2\gamma_1 k_1 \omega' -k_1^2} = 
                \sqrt{(\\alpha \\beta_{11} - \gamma_1^2) (\omega -\epsilon^2) 
                + 2\gamma_1 k_1 \omega -k_1^2 
                + \mathrm{j} 2 \epsilon 
                [\omega (\\alpha \\beta_{11} -\gamma_1^2) + \gamma_1 k_1]}` .
        
            Hence, if 
              
                :math:`\mathrm{sign}(\epsilon 
                [\omega (\\alpha \\beta_{11} -\gamma_1^2) + \gamma_1 k_1]) < 0`, 
              
            the imaginary part of :math:`k_3` becomes negative, and the 
            wavefield components :math:`\mathrm{e}^{\mathrm{j}k_3 x_3}` become 
            unstable. I fix that by manually modifying the vertical wavenumber 
            to 
    
                :math:`k_{3,mod} = \sqrt{(\\alpha \\beta_{11} -\gamma_1^2) 
                (\omega - \epsilon^2) + 2\gamma_1 k_1 \omega -k_1^2 
                + \mathrm{j} 2 \Vert \epsilon 
                [\omega (\\alpha \\beta_{11} -\gamma_1^2) + \gamma_1 k_1] \Vert}` .
    
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
        energy conservation does not hold for evanescent waves. (Reciprocal 
        media are not yet included in version 2.0. However, when reciprocal
        media are included this remark should be checked carefully.)
            
        
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
        
        \t:math:`k_3=-j \sqrt{ k_1^2 - 2\gamma_1 k_1 \omega - (\\alpha \\beta_{11} -\gamma_1^2)  \omega^2}`.
        
    - By default, **NumPy** makes the oppostie sign choice, 
        
        \t:math:`k_3=+j \sqrt{ k_1^2 - 2\gamma_1 k_1 \omega - (\\alpha \\beta_{11} -\gamma_1^2)\omega^2}`.
        
    - For convenience, we stick to **NumPy**'s sign choice. Thus, we will also adapt the sign choice for the propagators,
        
        - Kees chose: :math:`\\tilde{w}^{\pm} = \mathrm{exp}(\pm \lambda^{\pm} \Delta x_3)`.
        - We choose: :math:`\\tilde{w}^{\pm} = \mathrm{exp}(\mp \lambda^{\pm} \Delta x_3)`.
        
    References
    ----------
    
    Kees document as soon as it is published.
    
     
    Examples
    --------

    >>> from Layered_NRM_k1_w import Layered_NRM_k1_w as LM
    >>> import numpy as np
    
    >>> # Initialise wavefield in a layered non-reciprocal medium
    >>> F=LM(nt=1024,dt=0.005,nr=512,dx1=12.5,
    >>>      x3vec=np.array([1.1,2.2,3.7]), avec=np.array([1,2,3])*1e-3, 
    >>>      b11vec=np.array([1.4,3.14,2])*1e-4, 
    >>>      b13vec=np.array([0.4,2.4,1.2])*1e-4,
    >>>      b33vec=np.array([1.4,3.14,2])*1e-4,
    >>>      g1vec=np.array([0.8,2,1.3])*1e-4,
    >>>      g3vec=np.array([1.8,0.7,2.3])*1e-4,
    >>>      ReciprocalMedium=False)
    
    >>> # Get a meshgrid of the vertical-wavenumber
    >>> F.K3.shape
    (513, 512, 3)
    
    >>> # Get the vertical-wavenumber for omega=delta omega,  and k1=0
    >>> F.K3[1,0,0]
    (0.00044855235013677386+0j)
    
    **Wavefield Quantities**
    (Do not change the table!)
    
    +----------------------------------------+----------------------------+------------------------+------------------------------------------+-----------------------------------------+------------------------------------+
    |                                        | *TE*                       | *TM*                   | *Ac. (meta)*                             | *Ac. (rotat)*                           | *SH (solid)*                       |
    +========================================+============================+========================+==========================================+=========================================+====================================+
    | **P**                                  | :math:`E_2`                | :math:`H_2`            | :math:`\sigma`                           | :math:`\sigma`                          | :math:`v_2`                        |
    +----------------------------------------+----------------------------+------------------------+------------------------------------------+-----------------------------------------+------------------------------------+
    | :math:`\mathbf{Q}_1`                   | :math:`H_3`                | :math:`-E_3`           | :math:`v_1`                              | :math:`v_1`                             | :math:`-\\tau_{21}`                 |
    +----------------------------------------+----------------------------+------------------------+------------------------------------------+-----------------------------------------+------------------------------------+
    | :math:`\mathbf{Q}_3`                   | :math:`-H_1`               | :math:`E_1`            | :math:`v_3`                              | :math:`v_3`                             | :math:`-\\tau_{23}`                 |
    +----------------------------------------+----------------------------+------------------------+------------------------------------------+-----------------------------------------+------------------------------------+
    | :math:`\\boldsymbol{\\alpha}`            | :math:`\epsilon_{22}`      | :math:`\mu_{22}`       | :math:`\\frac{1}{K}`                      | :math:`\kappa`                          | :math:`\mathcal{R}_{22}`           |
    +----------------------------------------+----------------------------+------------------------+------------------------------------------+-----------------------------------------+------------------------------------+
    | :math:`\\boldsymbol{\\beta}_{11}`        | :math:`\mu_{33}`           | :math:`\epsilon_{33}`  | :math:`\mathcal{R}_{11}`                 | :math:`\\rho`                            | :math:`4 s_{1221}`                 |
    +----------------------------------------+----------------------------+------------------------+------------------------------------------+-----------------------------------------+------------------------------------+
    | :math:`\\boldsymbol{\\beta}_{13}`        | :math:`-\mu_{31}`          | :math:`-\epsilon_{31}` | :math:`\mathcal{R}_{13}`                 | :math:`\\frac{2\\rho\Omega_2}{j\omega}`   | :math:`4 s_{1223}`                 |
    +----------------------------------------+----------------------------+------------------------+------------------------------------------+-----------------------------------------+------------------------------------+
    | :math:`\\boldsymbol{\\beta}_{31}`        | :math:`-\mu_{13}`          | :math:`-\epsilon_{13}` | :math:`\mathcal{R}_{31}`                 | :math:`-\\frac{2\\rho\Omega_2}{j\omega}`  | :math:`4 s_{3221}`                 |
    +----------------------------------------+----------------------------+------------------------+------------------------------------------+-----------------------------------------+------------------------------------+
    | :math:`\\boldsymbol{\\beta}_{33}`        | :math:`\mu_{11}`           | :math:`\epsilon_{11}`  | :math:`\mathcal{R}_{33}`                 | :math:`\\rho`                            | :math:`4 s_{3223}`                 |
    +----------------------------------------+----------------------------+------------------------+------------------------------------------+-----------------------------------------+------------------------------------+
    | :math:`\\boldsymbol{\gamma}_{1}`        |  :math:`\\xi_{23}`          | :math:`-\zeta_{23}`    | :math:`\\frac{\\theta_{mm1}}{3K}`          |                                         |  :math:`-2 \\xi_{221}`              |
    +----------------------------------------+----------------------------+------------------------+------------------------------------------+-----------------------------------------+------------------------------------+
    | :math:`\\boldsymbol{\gamma}_{3}`        |  :math:`-\\xi_{21}`         | :math:`\zeta_{21}`     | :math:`\\frac{\\theta_{mm3}}{3K}`          |                                         |  :math:`-2 \\xi_{223}`              |
    +----------------------------------------+----------------------------+------------------------+------------------------------------------+-----------------------------------------+------------------------------------+
    | :math:`\\boldsymbol{\\chi}_{1}`          |  :math:`\zeta_{32}`        | :math:`-\\xi_{32}`      | :math:`-\\frac{\eta_{1ll}}{3K}`           |                                         | :math:`-2 \zeta_{122}`             |
    +----------------------------------------+----------------------------+------------------------+------------------------------------------+-----------------------------------------+------------------------------------+
    | :math:`\\boldsymbol{\\chi}_{3}`          |  :math:`-\zeta_{12}`       | :math:`\\xi_{12}`       | :math:`-\\frac{\eta_{3ll}}{3K}`           |                                         | :math:`-2 \zeta_{322}`             |
    +----------------------------------------+----------------------------+------------------------+------------------------------------------+-----------------------------------------+------------------------------------+
    | **B**                                  | :math:`-J_2^e`             | :math:`-J_2^m`         | :math:`q`                                | :math:`q`                               | :math:`f_2`                        |
    +----------------------------------------+----------------------------+------------------------+------------------------------------------+-----------------------------------------+------------------------------------+
    | :math:`\mathbf{C}_1`                   | :math:`-J_3^m`             | :math:`J_3^e`          | :math:`f_1`                              | :math:`f_1`                             | :math:`2h_{12}`                    |
    +----------------------------------------+----------------------------+------------------------+------------------------------------------+-----------------------------------------+------------------------------------+
    | :math:`\mathbf{C}_3`                   | :math:`J_1^m`              | :math:`-J_1^e`         | :math:`f_3`                              | :math:`f_3`                             | :math:`2h_{32}`                    |
    +----------------------------------------+----------------------------+------------------------+------------------------------------------+-----------------------------------------+------------------------------------+
    

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
        if b13vec.all() == np.zeros(1):
            b13vec = np.zeros_like(avec)
        if g1vec.all() == np.zeros(1):
            g1vec = np.zeros_like(avec)
        if g3vec.all() == np.zeros(1):
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
        self.eps = eps
        
        if (self.eps is not None) and (self.eps < 0):
            print('WARNING: Layered_NRM_k1_w\n'+72*'-'
                  +'\nThe parameter \'eps\' should be positive. \n'
                  +'(Exception) The function \'FocusingFunction_k1_w\' uses a '
                  +'negative \'eps\' value to handle the wrap-around correctly'
                  +' in case of complex-conjugated focusing functions.')
        
        # Calculate vertical wavenumber K3=K3(+k1) K3n=K3(-k1) 
        # Note: By default python uses opposite sign convention for evanescent 
        # waves as Kees: (-1)**0.5=1j
        K3  = np.zeros((self.nf,self.nr,self.x3vec.size),dtype=complex)
        K3n = K3.copy()
        W   = self.W_K1_grid()['Wgrid']
        K1  = self.W_K1_grid()['K1gridfft']
        if self.ReciprocalMedium is True:
            tmp = self.avec*self.b11vec + self.g1vec**2 + self.g3vec**2
            for layer in range(self.x3vec.size):
                K3[:,:,layer] = tmp[layer]*W**2 - K1**2
            K3n = K3.copy()
        elif self.ReciprocalMedium is False:
            tmp = self.avec*self.b11vec - self.g1vec**2
            if self.eps is None:
                for layer in range(self.x3vec.size):
                    K3[:,:,layer]  = tmp[layer]*W**2 + 2*self.g1vec[layer]*K1*W - K1**2
                    K3n[:,:,layer] = tmp[layer]*W**2 - 2*self.g1vec[layer]*K1*W - K1**2
                    
            # This is a manual fix to avoid exponential growth of the wavefield
            # I believe this fix sign-inverts epsilon for the unstable
            # w-k1 components. I am not sure if that is a problem. 
            # We account for the exceptional case where eps is negative (needed
            # for focusing functions) by inserting the signum term
            else:
                for layer in range(self.x3vec.size):
                    K3[:,:,layer]  = (tmp[layer]*W**2 
                                      + 2*self.g1vec[layer]*K1*W - K1**2)
                    K3n[:,:,layer] = (tmp[layer]*W**2 
                                       - 2*self.g1vec[layer]*K1*W - K1**2)
                    
                    K3[:,:,layer]  = (K3[:,:,layer].real  
                            + 1j*np.abs(K3[:,:,layer].imag) *np.sign(self.eps))
                    K3n[:,:,layer] = (K3n[:,:,layer].real 
                            + 1j*np.abs(K3n[:,:,layer].imag)*np.sign(self.eps))
                    
        self.K3  = K3**0.5
        self.K3n = K3n**0.5
        
        # Predefine eigenvalues
        self.LP  = None
        self.LM  = None
        self.LPn = None
        self.LMn = None
        
    
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
        
        >>> from Layered_NRM_k1_w import Layered_NRM_k1_w as LM
        >>> import numpy as np
        >>> F=LM(nt=1024,dt=0.005,nr=512,dx1=12.5,x3vec=np.array([1.1,2.2,3.7]),
                 avec=np.array([1,2,3])*1e-3,b11vec=np.array([1.4,3.14,2])*1e-4,
                 b13vec=np.array([0.4,2.4,1.2])*1e-4,b33vec=np.array([1.4,3.14,2])*1e-4,
                 g1vec=np.array([0.8,2,1.3])*1e-4,g3vec=np.array([1.8,0.7,2.3])*1e-4,
                 ReciprocalMedium=False)
        
        >>> # Create fk mask with a cut-off frequency at 200 1/s
        >>> Mask=F.FK1_mask_k1_w(wmax=200)
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
            tmp = self.avec*self.b11vec - self.g1vec**2
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
        
    def Eigenvalues_k1_w(self):
        """computes the eigenvalues :math:`\lambda^{\pm}` of the two-way 
        operator matrix :math:`\\rm{\\bf A}` in the wavenumber-frequency 
        domain. The eigenvalues are associated with the medium parameters 
        defined at initialisation of the wavefield.
            
            
        Returns
        -------
    
        dict
            Dictionary that contains 
            
                - **LP**: The eigenvalue :math:`\lambda^{+}`.
                - **LM**: The eigenvalue :math:`\lambda^{-}`.
                - **LPn**: The eigenvalue :math:`\lambda^{+}` with sign-inverted horizontal-wavenumbers :math:`k_1` (for adjoint medium). 
                - **LMn**: The eigenvalue :math:`\lambda^{-}` with sign-inverted horizontal-wavenumbers :math:`k_1` (for adjoint medium).   
                
            All eigenvalue matrices are stored in a in a (nf,nr,n)-array. The 
            dimensions correspond to the temporal frequencies :math:`\omega`, 
            the horizontal-wavenumbers :math:`k_1` and to the layers of the 
            medium.
        
        
        .. todo::
            
            (1) Check if the eigenvalues have to be modified for reciprocal 
            media.
            
            (2) Check if :math:`\\beta_{11} \\beta_{33} - \\beta_{13}^2 \geq 0` 
            holds in general.
        
        
        Notes
        -----
            
        - The eigenvalues are associated with non-reciprocal media.
        - We keep the sign-convention by Kees for the eigenvalues,
        
        :math:`\lambda^{\pm} = -j\omega \left(\gamma_3 + a_0 (\gamma_1-p_1)
        \pm \\sqrt{b_0b_1}p_3\\right)`.
            
        
        References
        ----------
        Kees document as soon as it is published.


        Examples
        --------

        >>> from Layered_NRM_k1_w import Layered_NRM_k1_w as LM
        >>> import numpy as np
        >>> F=LM(nt=1024,dt=0.005,nr=512,dx1=12.5,x3vec=np.array([1.1,2.2,3.7]),
                 avec=np.array([1,2,3])*1e-3,b11vec=np.array([1.4,3.14,2])*1e-4,
                 b13vec=np.array([0.4,2.4,1.2])*1e-4,b33vec=np.array([1.4,3.14,2])*1e-4,
                 g1vec=np.array([0.8,2,1.3])*1e-4,g3vec=np.array([1.8,0.7,2.3])*1e-4,
                 ReciprocalMedium=False)
        
        >>> # Compute the eigenvalue lambda_plus
        >>> Lam = F.Eigenvalues_k1_w()
        >>> lP = Lam["LP"]
        >>> lP[1,0,0]  # Evaluate for k1=0 and \omega = \Delta \omega
        -0.0006226976760655292j
        
        """
        # Frequency and horizontal-wavenumber meshgrids
        Om = self.W_K1_grid()['Wgrid']
        K1 = self.W_K1_grid()['K1gridfft']
        
        # Eigenvalues with sign-inverted horizontal-wavenumbers for adjoint 
        # medium
        LPn = None
        LMn = None
        
        if self.ReciprocalMedium is False:
            
            LP  = np.zeros((self.nf,self.nr,self.x3vec.size),dtype=complex)
            LM = LP.copy()
        
            for layer in range(self.x3vec.size):
                
                tmp1 = ( K1*self.b13vec[layer] 
                         + Om*(  self.b11vec[layer]*self.g3vec[layer] 
                               - self.b13vec[layer]*self.g1vec[layer] ) ) 
                        
                tmp2 = ( np.sqrt(  self.b11vec[layer]*self.b33vec[layer] 
                                 - self.b13vec[layer]**2 ) 
                        *self.K3[:,:,layer]*np.sign(self.b11vec[layer]) )
                
                LP[:,:,layer] = -1j*(tmp1 + tmp2) / self.b11vec[layer]
                LM[:,:,layer] = -1j*(tmp1 - tmp2) / self.b11vec[layer]
                    
            if self.AdjointMedium is True:
                LPn = LP.copy()
                LMn = LP.copy()
                
                for layer in range(self.x3vec.size):
                
                    tmp1 = ( -K1*self.b13vec[layer] 
                              + Om*(  self.b11vec[layer]*self.g3vec[layer] 
                                    - self.b13vec[layer]*self.g1vec[layer] ) ) 
                            
                    tmp2 = ( np.sqrt(  self.b11vec[layer]*self.b33vec[layer] 
                                     - self.b13vec[layer]**2 ) 
                            *self.K3n[:,:,layer]*np.sign(self.b11vec[layer]) )
                    
                    LPn[:,:,layer] = -1j*(tmp1 + tmp2) / self.b11vec[layer]
                    LMn[:,:,layer] = -1j*(tmp1 - tmp2) / self.b11vec[layer]
                
            if self.verbose is True:
                print('\nEigenvalues_k1_w (ReciprocalMedium is False)')
                print('---------------------------------------------')
                print('For non-reciprocal media, we compute the eigenvalues '
                      +'under the assumption that (b11*b33 - b13**2) is '
                      +'greater than, or equal to zero.')
        
        elif self.ReciprocalMedium is True:
            
            #####################
            # NOT YET IMPLEMENTED
            #####################
            if self.verbose is True:
                print('\nEigenvalues_k1_w (ReciprocalMedium is True)')
                print('--------------------------------------------')
                print('For reciprocal media, the eigenvalues are not yet '
                      +'implemented.')
                
        # Save eigenvalues in self
        self.LP  = LP
        self.LM  = LM
        self.LPn = LPn
        self.LMn = LMn
        
        return {"LP":LP,"LM":LM,"LPn":LPn,"LMn":LMn}
    
    
    def L_eigenvectors_k1_w(self,beta11=None,beta13=None,beta33=None,
                            K3=None,K3n=None,normalisation='flux'):
        """computes the eigenvector matrix 'L' and its inverse 'Linv', either 
        in flux- or in pressure-normalisation for the vertical-wavenumber 'K3' 
        inside a homogeneous layer. 
        
        Here, the vertical-wavenumber is a meshgrid that contains all 
        combinations of frequencies :math:`\omega` and horizontal-wavenumbers 
        :math:`k_1`. If \'AdjointMedium=True\', **L_eigenvectors_k1_w** also 
        computes the eigenvector matrix in the adjoint medium 'La' and its 
        inverse 'Lainv'. 
        
        Parameters
        ----------
    
        beta11 : int, float
            Medium parameter :math:`\\beta_{11}`  (real-valued).
        
        beta13 : int, float
            Medium parameter :math:`\\beta_{13}`  (real-valued).
            
        beta33 : int, float
            Medium parameter :math:`\\beta_{33}`  (real-valued).
        
        K3 : numpy.ndarray
            Vertical-wavenumber :math:`k_3(+k_1)` for all frequencies 
            :math:`\omega` and horizontal-wavenumbers :math:`k_1`.
        
        K3n : numpy.ndarray, optional (required if 'AdjointMedium=True')
            Vertical-wavenumber :math:`k_3(-k_1)` for all frequencies 
            :math:`\omega` and sign-inverted horizontal-wavenumbers :math:`k_1`.
            
        normalisation : str, optional
            For pressure-normalisation set normalisation='pressure', for 
            flux-normalisation set normalisation='flux'.
            
            
        Returns
        -------
    
        dict
            Dictionary that contains 
            
                - **L**: The eigenvector matrix.
                - **Linv**: The inverse of the eigenvector matrix.
                - **La**: The eigenvector matrix (adjoint medium).
                - **Lainv**: The inverse of the eigenvector matrix (adjoint medium).
                
            All eigenvector matrices are stored in a in a (nf,nr,2,2)-array. 
            The first two dimensions correspond to all combinations of 
            frequencies :math:`\omega` and horizontal-wavenumbers :math:`k_1`. 
            The last two dimension are the actual eigenvector matrices for all 
            :math:`\omega`-:math:`k_1` components.
        
        
        Notes
        -----
            
        - The eigenvector matrix 'L' and its inverse 'Linv' are different for reciprocal and non-reciprocal media (I assume that has not changed, but check it!).
        - For reciprocal media, the eigenvectors of the adjoint medium are identical to the eigenvectors of the true medium (verify!).
        - We have defined the eigenvectors of the adjoint medium only for flux-normalisation.
        - If the frequency :math:`\omega` is real-valued ('eps'=0): At zero frequency (:math:`\omega=0 \;\mathrm{s}^{-1}`), the eigenvector matrices \'L\' and their inverse \'Linv\' contain elements with poles. For computational convenience, we set the poles equal to zero. However, the resulting zero frequency component of all outputs is meaningless.
        
        
        References
        ----------
        Kees document as soon as it is published.


        Examples
        --------

        >>> from Layered_NRM_k1_w import Layered_NRM_k1_w as LM
        >>> import numpy as np
        
        >>> # Initialise wavefield in a layered non-reciprocal medium
        >>> F=LM(nt=1024,dt=0.005,nr=512,dx1=12.5,x3vec=np.array([1.1,2.2,3.7]),
        >>>      avec=np.array([1,2,3])*1e-3,b11vec=np.array([1.4,3.14,2])*1e-4,
        >>>      b13vec=np.array([0.4,2.4,1.2])*1e-4,
        >>>      b33vec=np.array([1.4,3.14,2])*1e-4,
        >>>      g1vec=np.array([0.8,2,1.3])*1e-4,
        >>>      g3vec=np.array([1.8,0.7,2.3])*1e-4,
        >>>      ReciprocalMedium=False,AdjointMedium=True)
        
        >>> # Compute eigenvectors in pressure-normalisation
        >>> Leig=F.L_eigenvectors_k1_w(beta11=F.b11vec[0],beta13=F.b13vec[0],
        >>>                            beta33=F.b33vec[0],K3=F.K3[:,:,0],
        >>>                            K3n=F.K3n[:,:,0],
        >>>                            normalisation='pressure')
        >>> L=Leig['L']
        
        >>> # For pressure normalisation, the top-left element of L equals
        >>> # 1 for all frequencies and all horizontal-wavenumbers
        >>> L[101,200,0,0]
        (1+0j)
        
        >>> # For pressure normalisation, the bottom-left element of L does not
        >>> # equal 1 for all frequencies and all horizontal-wavenumbers
        >>> L[101,200,1,0]
        10.85892500293106j
        
        """
        # Check if required input variables are given
        if ( (beta11 is None) or (beta13 is None) or (beta33 is None) 
             or (K3 is None) ):
            sys.exit('L_eigenvectors_k1_w: The input variables \'beta11\', '
                     +'\'beta13\', \'beta33\' and  \'K3\' must be set.')
         
        # Check if normalisation is set correctly
        if (normalisation is not 'flux') and (normalisation is not 'pressure'):
            sys.exit('L_eigenvectors_k1_w: The input variable '
                     +'\'normalisation\' must be set, either to \'flux\', or '
                     +'to \'pressure\'.')
            
        # Check if the vertical-wavenumber for a negative horizontal-wavenumber 
        # is given
        if ( (self.AdjointMedium is True) and (K3n is None) 
          and (normalisation is 'flux') and (self.ReciprocalMedium is False) ):
            sys.exit('L_eigenvectors_k1_w: The input variable \'K3n\' '
                     +'(vertical-wavenumber K3 for a negative horizontal-'
                     +'wavenumber k1) must be given to compute the eigenvector'
                     +' matrix of the adjoint medium \'La\' and its inverse '
                     +'\'Lainv\'.')
            
        # Initialise L and Linv
        L     = np.zeros((self.nf,self.nr,2,2),dtype=complex)
        Linv  = np.zeros((self.nf,self.nr,2,2),dtype=complex)
        La    = None
        Lainv = None
        
        # Construct a vertical ray-parameter
        Om = self.W_K1_grid()['Wgrid']
        P3 = K3.copy()
        
        # Exclude poles at zero-frequency if omega is real-valued
        if (self.eps is None) or (self.eps == 0):
            P3[1:,:] = K3[1:,:]/Om[1:,:]
            P3[0,:]  = 0
        else:
            P3 = K3/Om
            
        # Construct an inverse vertical ray-parameter
        # Exclude pole at zero frequency and zero horizontal-wavenumber
        # K3[0,0] = 0
        # P3[0,:] = 0
        P3inv = P3.copy()
        P3inv[1:,:] = Om[1:,:]/K3[1:,:]
        
        hinv = np.sqrt( beta11*beta33-beta13*beta13 )
        
        if self.verbose is True:
            print('\nL_eigenvectors_k1_w ')
            print(72*'-')
            print('The eigenvectors L and their inverse Linv have a pole at '
                  +'zero frequency. Here, we set the zero frequency component '
                  +'of L and Linv to zero (which is wrong but convenient for '
                  +'the computation).\n')

            
        if (self.ReciprocalMedium is True) and (normalisation is 'flux'):
            
            #####################
            # NOT YET IMPLEMENTED
            #####################
            if self.verbose is True:
                print('\nL_eigenvectors_k1_w (ReciprocalMedium is True)')
                print('-----------------------------------------------')
                print('For reciprocal media, the eigenvectors are not yet '
                      +'implemented.')
            
#            # L matrix
#            fac = (beta*P3inv/2)**0.5
#            L[:,:,0,0] = 1*fac
#            L[:,:,0,1] = 1*fac
#            L[:,:,1,0] = (P3+g3)/beta*fac
#            L[:,:,1,1] = -(P3-g3)/beta*fac
#            
#            # Inverse L matrix
#            Linv[:,:,0,0] = -L[:,:,1,1]
#            Linv[:,:,0,1] =  L[:,:,0,0]
#            Linv[:,:,1,0] =  L[:,:,1,0]
#            Linv[:,:,1,1] = -L[:,:,0,0]
#            
#            if self.AdjointMedium is True:
#                if self.verbose is True:
#                    print('\nL_eigenvectors_k1_w (AdjointMedium is True) and (ReciprocalMedium is True)')
#                    print('-------------------------------------------------------------------------------')
#                    print('For reciprocal media, the eigenvector matrix of a medium and its adjoint medium ')
#                    print('are identical.\n')
#               
#                La = L.copy()
#                Lainv = Linv.copy()
#            
#        elif (self.ReciprocalMedium is True) and (normalisation is 'pressure'):            
#            # L matrix
#            L[:,:,0,0] = 1
#            L[:,:,0,1] = 1
#            L[:,:,1,0] = (P3+g3)/beta
#            L[:,:,1,1] = -(P3-g3)/beta
#            
#            # Inverse L matrix
#            fac = beta*P3inv/2
#            Linv[:,:,0,0] = -L[:,:,1,1]*fac
#            Linv[:,:,0,1] = 1*fac
#            Linv[:,:,1,0] = L[:,:,1,0]*fac
#            Linv[:,:,1,1] = -1*fac
#            
#            if self.verbose is True and self.AdjointMedium is True:
#                print('\nL_eigenvectors_k1_w (AdjointMedium is True) and (normalisation=\'pressure\')')
#                print('-------------------------------------------------------------------------------')
#                print('We have defined the eigenvector matrix of the adjoint medium \'La\' and its ')
#                print('inverse \'Lainv\' only for flux-normalisation.\n')
            
        elif (self.ReciprocalMedium is False) and (normalisation is 'flux'):
            # L matrix
            L[:,:,0,0] = (hinv*P3inv)**0.5
            L[:,:,0,1] = L[:,:,0,0]
            L[:,:,1,0] = (P3/hinv)**0.5
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
                Om  = self.W_K1_grid()['Wgrid']
                P3n = K3n.copy()
                
                # Exclude poles at zero-frequency if omega is real-valued
                if (self.eps is None) or (self.eps == 0):
                    P3n[1:,:] = K3n[1:,:]/Om[1:,:]
                    P3n[0,:]  = 0
                else:
                    P3n = K3n/Om
                    
                # Construct an inverse vertical ray-parameter  for sign-inverted 
                # horizontal-wavenumbers
                # Exclude pole at zero frequency and zero horizontal-wavenumber
                # K3[0,0] = 0
                # P3[0,:] = 0
                P3ninv = P3n.copy()
                
                # Exclude poles at zero-frequency if omega is real-valued
                if (self.eps is None) or (self.eps == 0):
                    P3ninv[1:,:] = Om[1:,:]/K3n[1:,:]
                else:
                    P3ninv = Om/K3n
                
                # L matrix (adjoint medium) = N Transpose( Inverse( L(-k1) )) N
                La = np.zeros((self.nf,self.nr,2,2),dtype=complex)
                La[:,:,0,0] = (hinv*P3ninv)**0.5
                La[:,:,0,1] = La[:,:,0,0]
                La[:,:,1,0] = (P3n/hinv)**0.5
                La[:,:,1,1] = -La[:,:,1,0]
                La = La/2**0.5
                
                #  Inverse L matrix (adjoint medium) = N Transpose( L(-k1)) N
                Lainv = np.zeros((self.nf,self.nr,2,2),dtype=complex)
                Lainv[:,:,0,0] = La[:,:,1,0]
                Lainv[:,:,0,1] = La[:,:,0,0]
                Lainv[:,:,1,0] = La[:,:,1,0]
                Lainv[:,:,1,1] = - La[:,:,0,0]
            
        elif (self.ReciprocalMedium is False) and (normalisation is 'pressure'):
            # L matrix
            L[:,:,0,0] = 1
            L[:,:,0,1] = 1
            L[:,:,1,0] = P3/hinv
            L[:,:,1,1] = -L[:,:,1,0]
            
            # Inverse L matrix
            Linv[:,:,0,0] = 1
            Linv[:,:,0,1] = hinv*P3inv
            Linv[:,:,1,0] = 1
            Linv[:,:,1,1] = -hinv*P3inv
            Linv = 0.5*Linv
            
            if self.verbose is True and self.AdjointMedium is True:
                print('\nL_eigenvectors_k1_w (AdjointMedium is True) and (normalisation=\'pressure\')')
                print('-------------------------------------------------------------------------------')
                print('We have defined the eigenvector matrix of the adjoint medium \'La\' and its ')
                print('inverse \'Lainv\' only for flux-normalisation.\n')
        
        out = {'L':L,'Linv':Linv,'La':La,'Lainv':Lainv}
        return out
          
    def RT_k1_w(self,beta11_u=None,beta13_u=None,beta33_u=None,K3_u=None,K3n_u=None,
                     beta11_l=None,beta13_l=None,beta33_l=None,K3_l=None,K3n_l=None,
                     normalisation='flux'):
        """computes the scattering coefficients at an horizontal interface.
        
        The scattering coefficients can be computed either in flux- or in 
        pressure-normalisation. The variables with subscript 'u' refer to the 
        medium parameters in the upper half-space, the variables with subscript 
        'l' refer to the medium parameters in the lower half-space. The 
        vertical-wavenumbers \'K3\':math:`=k_3(k_1,\omega)` and 
        \'K3n\':math:`=k_3(-k_1,\omega)` are stored as :math:`k_1`-
        :math:`\omega` meshgirds to compute the scattering coefficients for all 
        sampled frequencies and horizontal-wavenumbers in a vectorsied manner. 
        Set \'AdjointMedium=True\' to compute the scattering coefficients also 
        in the adjoint medium.
        
        Parameters
        ----------
    
        beta11_u : int, float
            Medium parameter :math:`\\beta_{11}` (real-valued) (upper half-space).
        
        beta13_u : int, float
            Medium parameter :math:`\\beta_{13}` (real-valued) (upper half-space).
            
        beta33_u : int, float
            Medium parameter :math:`\\beta_{33}` (real-valued) (upper half-space).
        
        K3_u : numpy.ndarray
            Vertical-wavenumber :math:`k_{3,u}(+k_1)` for all frequencies 
            :math:`\omega` and horizontal-wavenumbers :math:`k_1` (upper half-
            space).
        
        K3n_u : numpy.ndarray, optional (required if 'AdjointMedium=True')
            Vertical-wavenumber :math:`k_{3,u}(-k_1)` for all frequencies 
            :math:`\omega` and sign-inverted horizontal-wavenumbers :math:`k_1` 
            (upper half-space).
            
        beta11_l : int, float
            Medium parameter :math:`\\beta_{11}` (real-valued) (lower half-space).
        
        beta13_l : int, float
            Medium parameter :math:`\\beta_{13}` (real-valued) (lower half-space).
            
        beta33_l : int, float
            Medium parameter :math:`\\beta_{33}` (real-valued) (lower half-space).
        
        K3_l : numpy.ndarray
            Vertical-wavenumber :math:`k_{3,l}(+k_1)` for all frequencies 
            :math:`\omega` and horizontal-wavenumbers :math:`k_1` (lower half-
            space).
        
        K3n_l : numpy.ndarray, optional (required if 'AdjointMedium=True')
            Vertical-wavenumber :math:`k_{3,l}(-k_1)` for all frequencies 
            :math:`\omega` and sign-inverted horizontal-wavenumbers :math:`k_1` 
            (lower half-space).
            
        normalisation : str, optional
            For pressure-normalisation set normalisation='pressure', for flux-
            normalisation set normalisation='flux'.
            
        Returns
        -------
    
        dict
            Dictionary that contains 
            
                - **rP**: Reflection coefficient from above.
                - **tP**: Transmission coefficient from above.
                - **rM**: Reflection coefficient from below.
                - **tM**: Transmission coefficient from below.
                - **rPa**: Reflection coefficient from above (adjoint medium).
                - **tPa**: Transmission coefficient from above (adjoint medium).
                - **rMa**: Reflection coefficient from below (adjoint medium).
                - **tMa**: Transmission coefficient from below (adjoint medium).
                
            All scattering coefficients are stored as arrays with the shape 
            (nf,nr).
            
        
        .. todo::
            
            (1) For :math:`(k_1 , \omega) = (0,0)` there is a zero division in the 
            computation of the scattering coefficients. I have fixed that, however, I believe that the fix is (mathmatically) wrong. Check that! If the frequency is complex-valued, :math:`\omega'=\omega+j\epsilon`, the zero division is omitted and there is no manual fix.
        
        
        Notes
        -----
    
        - For reciprocal media, the scattering coefficients of the adjoint medium are identical to the scattering coefficients of the true medium. (To be checked)
        
            
        References
        ----------
        
        Kees document as soon as it is published.
        
        
        Examples
        --------
    
        >>> from Layered_NRM_k1_w import Layered_NRM_k1_w as LM
        >>> import numpy as np
        
        >>> # Initialise wavefield F in a reciprocal medium 
        >>> F=LM(nt=1024,dt=0.005,nr=512,dx1=12.5,x3vec=np.array([1.1,2.2,3.7]),
        >>>      avec=np.array([1,2,3])*1e-3,b11vec=np.array([1.4,3.14,2])*1e-4,
        >>>      b13vec=np.array([0.4,2.4,1.2])*1e-4,b33vec=np.array([1.4,3.14,2])*1e-4,
        >>>      g1vec=np.array([0.8,2,1.3])*1e-4,g3vec=np.array([1.8,0.7,2.3])*1e-4,
        >>>      ReciprocalMedium=False,AdjointMedium=True)
        
        >>> # Compute scattering coefficients at the first interface in flux
        >>> # normalisation
        >>> Scat = F.RT_k1_w(beta11_u=F.b11vec[0],beta13_u=F.b13vec[0],beta33_u=F.b33vec[0],
        >>>                  K3_u=F.K3[:,:,0],K3n_u=F.K3n[:,:,0],
        >>>                  beta11_l=F.b11vec[1],beta13_l=F.b13vec[1],beta33_l=F.b33vec[1],
        >>>                  K3_l=F.K3[:,:,1],K3n_l=F.K3n[:,:,1],normalisation='flux')
        
        >>> # Read the scattering coeffcients, and 
        >>> tP = Scat['tP']
        >>> rM = Scat['rM']
        >>> rP = Scat['rP']
        >>> tM = Scat['tM'] 
        
        >>> tP.shape
        (513, 512)
        
        >>> np.linalg.norm(tP-tM)
        0.0
        
        >>> # Transmission coefficient for k1,omega = (0,Delta omega)
        >>> tP[1,0]
        (0.9865896281519458+0j)
        
        
        """
        
        # Check if required input variables are given
        if (   (beta11_u is None) or (beta13_u is None) or (beta33_u is None) 
            or (beta11_l is None) or (beta13_l is None) or (beta33_l is None) 
            or (K3_u is None) or (K3_l is None)):
            sys.exit('RT_k1_w: The input variables \'beta11_u\', \'beta13_u\','
                     +' \'beta33_u\', \'K3_u\', \'beta11_l\', \'beta13_l\', '
                     +'\'beta33_l\',  \'K3_l\' must be set.')
            
        # Check if normalisation is set correctly
        if (normalisation is not 'flux') and (normalisation is not 'pressure'):
            sys.exit('RT_k1_w: The input variable \'normalisation\' must be'+
                     ' set, either to \'flux\', or to \'pressure\'.')
            
        # Check if the vertical-wavenumber for a sign-inverted horizontal-
        # wavenumber is given
        if  (    (self.AdjointMedium is True) 
             and ((K3n_u is None) or (K3n_l is None))  
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
        
        # For zero frequency and zero horizontal-wavenumber we will encounter
        # divisions by zero. 
        # To avoid this problem we modify the (w,k1)=(0,0) element of K3,
        # such that there is no division by zero, and such that the resulting
        # scattering coefficients are correct
        # The (w,k1)=(0,0) element of K3 is actually a ray-parameter P3
        if (self.eps is None) or (self.eps == 0):
            K3_u[0,0]  = K3_u[1,0]/self.Dw()
            K3_l[0,0]  = K3_l[1,0]/self.Dw()
            K3n_u[0,0] = K3n_u[1,0]/self.Dw()
            K3n_l[0,0] = K3n_l[1,0]/self.Dw()
        
        hu = 1/np.sqrt( beta11_u*beta33_u-beta13_u*beta13_u )
        hl = 1/np.sqrt( beta11_l*beta33_l-beta13_l*beta13_l )
        
        if (self.ReciprocalMedium is True) and (normalisation is 'flux'):
            
            #####################
            # NOT YET IMPLEMENTED
            #####################
            if self.verbose is True:
                print('\nRT_k1_w (ReciprocalMedium is True)')
                print('-----------------------------------------------')
                print('For reciprocal media, the scattering coefficients are '
                      +'not yet implemented.')
            
#            # True medium
#            denom = 1/( (K3_u-Om*g3_u)*beta_l + (K3_l+Om*g3_l)*beta_u )
#            rP =  ( (K3_u+Om*g3_u)*beta_l - (K3_l+Om*g3_l)*beta_u ) * denom 
#            rM = -( (K3_u-Om*g3_u)*beta_l - (K3_l-Om*g3_l)*beta_u ) * denom
#            tP = 2*(K3_u*beta_l*K3_l*beta_u)**0.5 * denom
#            tM = tP
#            
#            # Correct the zero frequency, zero horizontal-wavenumber component
#            denom = 1/( (K3_u[0,0]-g3_u)*beta_l + (K3_l[0,0]+g3_l)*beta_u )
#            rP[0,0] =  ( (K3_u[0,0]+g3_u)*beta_l - (K3_l[0,0]+g3_l)*beta_u ) * denom 
#            rM[0,0] = -( (K3_u[0,0]-g3_u)*beta_l - (K3_l[0,0]-g3_l)*beta_u ) * denom
#            tP[0,0] = 2*(K3_u[0,0]*beta_l*K3_l[0,0]*beta_u)**0.5 * denom
#            tM[0,0] = tP[0,0]
#            
#            if self.AdjointMedium is True:
#                # Adjoint medium
#                rPa = rP 
#                tPa = tP 
#                rMa = rM 
#                tMa = tM 
#    
#                if self.verbose is True:
#                    print('\nRT_k1_w: (AdjointMedium is True) and (ReciprocalMedium is True)')
#                    print(72*'-')
#                    print('For reciprocal media, the scattering coefficients in a medium and its adjoint medium are identical.\n')
#        
#        elif (self.ReciprocalMedium is True) and (normalisation is 'pressure'):
#            
#            # True medium
#            denom = 1/( (K3_u-Om*g3_u)*beta_l + (K3_l+Om*g3_l)*beta_u )
#            rP =  ( (K3_u+Om*g3_u)*beta_l - (K3_l+Om*g3_l)*beta_u ) * denom
#            rM = -( (K3_u-Om*g3_u)*beta_l - (K3_l-Om*g3_l)*beta_u ) * denom
#            tP = 2*K3_u*beta_l * denom
#            tM = 2*K3_l*beta_u * denom
#            
#            # Correct the zero frequency, zero horizontal-wavenumber component
#            denom = 1/( (K3_u[0,0]-g3_u)*beta_l + (K3_l[0,0]+g3_l)*beta_u )
#            rP[0,0] =  ( (K3_u[0,0]+g3_u)*beta_l - (K3_l[0,0]+g3_l)*beta_u ) * denom 
#            rM[0,0] = -( (K3_u[0,0]-g3_u)*beta_l - (K3_l[0,0]-g3_l)*beta_u ) * denom
#            tP[0,0] = 2*K3_u[0,0]*beta_l * denom
#            tM[0,0] = 2*K3_l[0,0]*beta_u * denom
#            
#            if self.verbose is True and self.AdjointMedium is True:
#                print('\nRT_k1_w: (AdjointMedium is True) and (normalisation=\'pressure\')')
#                print(72*'-')
#                print('We have defined the scattering coefficients of the adjoint medium only for flux-normalisation.\n')
            
        elif (self.ReciprocalMedium is False) and (normalisation is 'flux'):
            
            # True medium (no need to coorect (w,k1)=(0,0) element)
            denom = 1/(K3_u*hu + K3_l*hl)
            rP = (K3_u*hu - K3_l*hl) *denom
            tP = 2*(K3_u*hu*K3_l*hl)**0.5 * denom
            rM = -rP
            tM = tP
            
            if self.AdjointMedium is True:
                # Adjoint medium (no need to coorect (w,k1)=(0,0) element)
                denom = 1/(K3n_u*hu + K3n_l*hl)
                rPa = (K3n_u*hu - K3n_l*hl) * denom
                tPa = 2*(K3n_u*hu*K3n_l*hl)**0.5 * denom
                rMa = -rPa
                tMa = tPa 
            
        elif (self.ReciprocalMedium is False) and (normalisation is 'pressure'):
            
            # True medium  (no need to coorect (w,k1)=(0,0) element)
            denom = 1/(K3_u*hu + K3_l*hl)
            rP = (K3_u*hu - K3_l*hl) * denom
            tP = 2*K3_u*hu*denom
            rM = -rP
            tM = 1 - rP
            
            if self.AdjointMedium is True:
                # Adjoint medium (no need to coorect (w,k1)=(0,0) element)
                denom = 1/(K3n_u*hu + K3n_l*hl)
                rPa = (K3n_u*hu - K3n_l*hl) * denom
                tPa = 2*K3n_u*hu*denom
                rMa = -rPa
                tMa = 1 - rPa 
            
        out = {'rP':rP,'tP':tP,'rM':rM,'tM':tM,'rPa':rPa,'tPa':tPa,'rMa':rMa,'tMa':tMa}
        return out
    
    def W_propagators_k1_w(self,LP=None,LM=None,LPn=None,LMn=None,dx3=None):
        """computes the downgoing propagator 'wP' and the upgoing progagator 
        'wM' for all sampled eigenvalues 'LP' and 'LM' and a vertical distance 
        'dx3' (downward pointing :math:`x_3`-axis).
        
        
        Parameters
        ----------
    
        LP : numpy.ndarray
            Eigenvalus :math:`\lambda^+` for all frquencies :math:`\omega` and 
            all horizontal-wavenumbers :math:`k_1`.
            
        LM : numpy.ndarray
            Eigenvalus :math:`\lambda^-` for all frquencies :math:`\omega` and 
            all horizontal-wavenumbers :math:`k_1`.
        
        LPn : numpy.ndarray, optional (required if 'AdjointMedium=True')
            Eigenvalus :math:`\lambda^+` for all frquencies :math:`\omega` and 
            all sign-inverted horizontal-wavenumbers :math:`k_1`.
            
        LMn : numpy.ndarray, optional (required if 'AdjointMedium=True')
            Eigenvalus :math:`\lambda^-` for all frquencies :math:`\omega` and 
            all sign-inverted horizontal-wavenumbers :math:`k_1`.
            
        dx3 : int, float
            Vertical propagation distance :math:`\Delta x_3` (downward 
            pointing :math:`x_3`-axis). The variable 'dx3' should be greater 
            than, or equal to zero.
            
        Returns
        -------
    
        dict
            Dictionary that contains 
            
                - **wP**: Downward propagator :math:`\\tilde{w}^+`.
                - **wM**: Upward propagator :math:`\\tilde{w}^-`.
                - **wPa**: Downward propagator :math:`\\tilde{w}^{+(a)}` (adjoint medium). 
                - **wMa**: Upward propagator :math:`\\tilde{w}^{-(a)}` (adjoint medium). 
                
            All propagators are stored in an arrays of shape (nf,nr). The 
            variables 'wPa' and 'wMa' are computed only for the setting 
            'AdjointMedium=True'.
        
        
        .. todo::
            
            In a non-reciprocal medium, for a complex-valued frequency 
            :math:`\omega'=\omega+\mathrm{j}\epsilon` one of the propagators 
            has an exponentially growing term. Does that cause errors? If yes, 
            can we fix that manually? (still the case?)
        
        
        Notes
        -----
        
        - For evanescent waves, Kees makes a sign choice for the vertical-wavenumber,
        
        \t:math:`k_3=-j \sqrt{ k_1^2 - 2\gamma_1 k_1 \omega - (\\alpha \\beta_{11} -\gamma_1^2)  \omega^2}`.
        
        - By default, **NumPy** makes the oppostie sign choice, 
        
        \t:math:`k_3=+j \sqrt{ k_1^2 - 2\gamma_1 k_1 \omega - (\\alpha \\beta_{11} -\gamma_1^2)\omega^2}`.
        
        - For convenience, we stick to **NumPy**'s sign choice. Thus, we will also adapt the sign choice for the propagators,
        
            - Kees chose: :math:`\\tilde{w}^{\pm} = \mathrm{exp}(\pm \lambda^{\pm} \Delta x_3)`.
            - We choose: :math:`\\tilde{w}^{\pm} = \mathrm{exp}(\mp \lambda^{\pm} \Delta x_3)`. 
        
        
        References
        ----------
        
        Kees document as soon as it is published.


        Examples
        --------
        
        >>> from Layered_NRM_k1_w import Layered_NRM_k1_w as LM
        >>> import numpy as np
        
        >>> # Initialise a wavefield
        >>> F=LM(nt=1024,dt=0.005,nr=512,dx1=12.5,x3vec=np.array([1.1,2.2,3.7]),
        >>>      avec=np.array([1,2,3])*1e-3,b11vec=np.array([1.4,3.14,2])*1e-4,
        >>>      b13vec=np.array([0.4,2.4,1.2])*1e-4,b33vec=np.array([1.4,3.14,2])*1e-4,
        >>>      g1vec=np.array([0.8,2,1.3])*1e-4,g3vec=np.array([1.8,0.7,2.3])*1e-4,
        >>>      ReciprocalMedium=False,AdjointMedium=True)

        >>> # Compute eigenvalues in layer 0
        >>> Eig = F.Eigenvalues_k1_w()
        >>> LP=Eig["LP"][:,:,0]
        >>> LM=Eig["LM"][:,:,0]
        >>> LPn=Eig["LPn"][:,:,0]
        >>> LMn=Eig["LMn"][:,:,0]
        
        >>> # Compute the propagators of the first layer
        >>> Prop = F.W_propagators_k1_w(LP=LP[:,:,0],LPn=LPn[:,:,0],
                                        LM=LM[:,:,0],LMn=LMn[:,:,0],
                                        dx3=F.x3vec[0])
        >>> wP = np.fft.fftshift(Prop['wP'],axes=1)
        >>> wM = np.fft.fftshift(Prop['wM'],axes=1)
        >>> wPa = np.fft.fftshift(Prop['wPa'],axes=1)
        >>> wMa = np.fft.fftshift(Prop['wMa'],axes=1)
        >>> # We plot the resulting propagators below.
        
        
        .. image:: ../pictures/cropped/Propagators.png
           :width: 200px
           :height: 200px
       
        
        """
        
        # Check if required input variables are given
        if (    (np.shape(LP) != (self.nf,self.nr)) 
             or (np.shape(LM) != (self.nf,self.nr)) ):
            sys.exit('W_propagators_k1_w: The input variables \'LP\' and  '
                     +'\'LM\' must be given and must have the shape (nf,nr).')
            
        # Check if dx3 is a scalar
        if ( (not np.isscalar(dx3)) or (not isinstance(dx3,float))
            or (not isinstance(dx3,float)) ):
            sys.exit('The input variable \'dx3\' must be given and must be a '
                     +'scalar.')
            
        # If AdjointMedium=True it is required to set LPn=LP(-k1) and LMn=LM(-k1)
        if (    (self.AdjointMedium is True) 
            and (np.shape(LPn) != (self.nf,self.nr))
            and (np.shape(LMn) != (self.nf,self.nr)) ):
            sys.exit('W_propagators_k1_w: If \'AdjointMedium=True\' the input '
                     +'variables \'LPn\' and \'LMn\' must be given, and they '
                     +'must have the shape (nf,nr).')
        
        # Propagators in the adjoint medium
        wPa = None
        wMa = None
        
        if self.ReciprocalMedium is True:
            #####################
            # NOT YET IMPLEMENTED
            #####################
            if self.verbose is True:
                print('\nW_propagators_k1_w (ReciprocalMedium is True)')
                print('-----------------------------------------------')
                print('For reciprocal media, the propagators are not yet '
                      +'implemented.')
#            wP = np.exp(1j*K3*dx3)
#            wM = wP.copy()
#            
#            if self.AdjointMedium is True:
#                wPa = np.exp(1j*K3n*dx3)
#                wMa = wPa.copy()                
        
        elif self.ReciprocalMedium is False:
            
            wP = np.exp(-LP*dx3)
            wM = np.exp(LM*dx3)
        
            if self.AdjointMedium is True:
                wPa = np.exp(LMn*dx3)
                wMa = np.exp(-LPn*dx3)
           
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
            
    
    def RT_response_k1_w(self,x3vec=None,avec=None,b11vec=None,b13vec=None,
                         b33vec=None,g1vec=None,g3vec=None,eps=None,
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
            
        b11vec : numpy.ndarray, optional
            Medium parameter :math:`\\beta_{11}` (real-valued), for n layers 
            'b11vec' must have the shape (n,).
            
        b13vec : numpy.ndarray, optional
            Medium parameter :math:`\\beta_{13}` (real-valued), for n layers 
            'b11vec' must have the shape (n,).
            
        b33vec : numpy.ndarray, optional
            Medium parameter :math:`\\beta_{33}` (real-valued), for n layers 
            'b11vec' must have the shape (n,).
        
        g1vec : numpy.ndarray, optional
            Medium parameter :math:`\gamma_1` (real-valued for non-reciprocal 
            media or imaginary-valued for reciprocal media), for n layers 
            'g1vec' must have the shape (n,).
            
        g3vec : numpy.ndarray, optional
            Medium parameter :math:`\gamma_3` (real-valued for non-reciprocal 
            media or imaginary-valued for reciprocal media), for n layers 
            'g3vec' must have the shape (n,).
            
        eps : int, float, optional
            A real-valued scalar can be assigned to 'eps' to reduce the wrap-
            around effect of wavefields in the time domain. The parameter 'eps' 
            (:math:`=\epsilon`) should be positive to the suppress wrap-around 
            effect from positive to negative time (Recommended value eps = 
            :math:`\\frac{3}{n_f dt}`). If not defined, the value 'self.eps' is
            used.
            
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
        >>> F = LM(LM(nt=1024,dt=0.005,nr=2048,dx1=12.5,
        >>>           x3vec=np.array([1.1,2.2,3.7,4])*4e2,
        >>>           avec=np.array([1,2,3,4])*1e-3,
        >>>           b11vec=np.array([1.4,3.14,2,4])*1e-4,
        >>>           b13vec=np.array([0.4,2.4,1.2,1.2])*1e-4,
        >>>           b33vec=np.array([1.4,3.14,2,2])*1e-4,
        >>>           g1vec=np.array([0.8,2,1.3,1.3])*1e-4,
        >>>           g3vec=np.array([1.8,0.7,2.3,2.3])*1e-4,
        >>>           eps=3/(513*0.005),ReciprocalMedium=False,
        >>>           AdjointMedium=True)
        
        >>> # Model the medium responses
        >>> RT = F.RT_response_k1_w(normalisation='flux',InternalMultiples=True)
        >>> RP = RT['RP']
        >>> TP = RT['TP']
        >>> RM = RT['RM']
        >>> TM = RT['TM']
        
        
        >>> # Make a Ricker wavelet
        >>> Wav=F.RickerWavelet_w(f0=30)
    
        >>> # Compute correction for complex-valued frequency
        >>> gain = np.fft.fftshift(Fe.Gain_t(),axes=0)
        
        >>> # Apply inverse Fourier transform to the space-time domain
        >>> tP = gain*np.fft.fftshift(Fe.K1W2X1T(Wav*TP),axes=(0,1))
        >>> rM = gain*np.fft.fftshift(Fe.K1W2X1T(Wav*RM),axes=(0,1))
        >>> rP = gain*np.fft.fftshift(Fe.K1W2X1T(Wav*RP),axes=(0,1))
        >>> tM = gain*np.fft.fftshift(Fe.K1W2X1T(Wav*TM),axes=(0,1))
        >>> # We plot the resulting wavefields below.
        
        .. image:: ../pictures/cropped/RT_Responses.png
            :height: 200px
            :width: 300 px
        
        
        """
        
        # Check if normalisation is set correctly
        if (normalisation is not 'flux') and (normalisation is not 'pressure'):
            sys.exit('RT_response_k1_w: The input variable \'normalisation\' '+
                     'must be set, either to \'flux\', or to \'pressure\'.')
        
        # Check if a layer stack is given
        if (isinstance(x3vec,np.ndarray) and isinstance(avec,np.ndarray) 
        and isinstance(b11vec,np.ndarray) and isinstance(b13vec,np.ndarray) 
        and isinstance(b33vec,np.ndarray) and isinstance(g1vec,np.ndarray) 
        and isinstance(g3vec,np.ndarray)):
            
            if eps is None:
                subeps = self.eps
            else:
                subeps = eps
            
            # Create a wavefield in a sub-medium
            # I do this because when the sub-wavefield is initialised all 
            # parameters are automatically tested for correctness
            self.SubSelf = Layered_NRM_k1_w(self.nt,self.dt,self.nr,self.dx1,
                                            self.verbose,eps=subeps,
                                            x3vec=x3vec,avec=avec,
                                            b11vec=b11vec,b13vec=b13vec,
                                            b33vec=b33vec,
                                            g1vec=g1vec,g3vec=g3vec,
                                            ReciprocalMedium=self.ReciprocalMedium,
                                            AdjointMedium=self.AdjointMedium)
            
            x3vec   = self.SubSelf.x3vec
            b11vec  = self.SubSelf.b11vec
            b13vec  = self.SubSelf.b13vec
            b33vec  = self.SubSelf.b33vec
            K3      = self.SubSelf.K3
            K3n     = self.SubSelf.K3n
            Eig     = self.SubSelf.Eigenvalues_k1_w()
                
        # Else compute response of entire medium
        else:
            x3vec   = self.x3vec
            b11vec  = self.b11vec
            b13vec  = self.b13vec
            b33vec  = self.b33vec
            K3      = self.K3
            K3n     = self.K3n    
            Eig     = self.Eigenvalues_k1_w()
        
        # Eigenvalues
        LP  = Eig['LP']
        LM  = Eig['LM']
        LPn = Eig['LPn']
        LMn = Eig['LMn']
        del Eig
        
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
            ScatCoeffs = self.RT_k1_w(beta11_u=b11vec[n],beta13_u=b13vec[n],
                                      beta33_u=b33vec[n],K3_u=K3[:,:,n],
                                      K3n_u=K3n[:,:,n],
                                      beta11_l=b11vec[n+1],beta13_l=b13vec[n+1],
                                      beta33_l=b33vec[n+1],K3_l=K3[:,:,n+1],
                                      K3n_l=K3n[:,:,n+1],
                                      normalisation=normalisation)
            rP = ScatCoeffs['rP']
            tP = ScatCoeffs['tP']
            rM = ScatCoeffs['rM']
            tM = ScatCoeffs['tM']
            
            # Propagators
            if self.AdjointMedium is True:
                W = self.W_propagators_k1_w(LP=LP[:,:,n],LM=LM[:,:,n],
                                            LPn=LPn[:,:,n],LMn=LMn[:,:,n],
                                            dx3=dx3vec[n])
            else:
                W = self.W_propagators_k1_w(LP=LP[:,:,n],LM=LM[:,:,n],
                                            dx3=dx3vec[n])
                
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
                - **b11vec**:  Updated :math:`\\beta_{11}` vector.
                - **b13vec**:  Updated :math:`\\beta_{11}` vector.
                - **b33vec**:  Updated :math:`\\beta_{11}` vector.
                - **g1vec**: Updated :math:`\gamma_1` vector.
                - **g3vec**: Updated :math:`\gamma_3` vector.
                - **K3**:    Updated :math:`k_3(k_1)` vector.
                - **K3n**:   Updated :math:`k_3(-k_1)` vector.
                - **LP**:    Updated eigenvalues :math:`\lambda^+(k_1)`.
                - **LPn**:   Updated eigenvalues :math:`\lambda^+(-k_1)`.
                - **LM**:    Updated eigenvalues :math:`\lambda^-(k_1)`.
                - **LMn**:   Updated eigenvalues :math:`\lambda^-(-k_1)`.
                
            All medium parameter vectors are stored in arrays of shape (n,).
            The vertical wavenumbers and the eigenvalues are stored in arrays 
            of shape (nf,nr,n).
        
        Examples
        --------
        
        >>> from Layered_NRM_k1_w import Layered_NRM_k1_w as LM
        >>> import numpy as np
        
        >>> # Initialise a wavefield in a 1D reciprocal medium
        >>> F = LM(LM(nt=1024,dt=0.005,nr=2048,dx1=12.5,
        >>>           x3vec=np.array([1.1,2.2,3.7,4])*4e2,
        >>>           avec=np.array([1,2,3,4])*1e-3,
        >>>           b11vec=np.array([1.4,3.14,2,4])*1e-4,
        >>>           b13vec=np.array([0.4,2.4,1.2,1.2])*1e-4,
        >>>           b33vec=np.array([1.4,3.14,2,2])*1e-4,
        >>>           g1vec=np.array([0.8,2,1.3,1.3])*1e-4,
        >>>           g3vec=np.array([1.8,0.7,2.3,2.3])*1e-4,
        >>>           eps=3/(513*0.005),ReciprocalMedium=False,
        >>>           AdjointMedium=True)
        
        >>> # Insert a transparent layer at x3=1
        >>> out=F.Insert_layer(x3=1,UpdateSelf=False)
        
        >>> # Updated depth vector
        >>> out['x3vec']
        array([1.00e+00, 4.40e+02, 8.80e+02, 1.48e+03, 1.60e+03])
        
        >>> # Updated alpha vector
        >>> out['avec']
        array([0.001, 0.001, 0.002, 0.003, 0.004])
        
        """
        
        # Check if x3 is a scalar or an array of the shape (n,).
        if not (    isinstance(x3,int) 
                or  isinstance(x3,float) 
                or (isinstance(x3,np.ndarray) and x3.ndim == 1) ):
            sys.exit('Insert_layer: The input variable \'x3\' must be either '
                     +'a real-valued scalar, or an array of shape (n,) with '
                     +'real-valued elements.')
        
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
        
        X3vec  = self.x3vec
        Avec   = self.avec
        B11vec = self.b11vec
        B13vec = self.b13vec
        B33vec = self.b33vec
        G1vec  = self.g1vec
        G3vec  = self.g3vec
        K3     = self.K3
        K3n    = self.K3n
        LP     = self.LP
        LPn    = self.LPn
        LM     = self.LM
        LMn    = self.LMn
        
        for i in range(np.size(x3)):
        
            # Vector of depths smaller than or equal to x3[i]
            L = X3vec[X3vec<=x3[i]] 
            
            # Case1: x3[i] smaller than X3vec[0]
            if L.size == 0:
                X3vec  = np.hstack([x3[i]      ,X3vec])
                Avec   = np.hstack([Avec[0]    ,Avec])
                B11vec = np.hstack([B11vec[0]  ,B11vec])
                B13vec = np.hstack([B13vec[0]  ,B13vec])
                B33vec = np.hstack([B33vec[0]  ,B33vec])
                G1vec  = np.hstack([G1vec[0]   ,G1vec])
                G3vec  = np.hstack([G3vec[0]   ,G3vec])
                K3     = np.dstack([K3[:,:,:1] ,K3])
                K3n    = np.dstack([K3n[:,:,:1],K3n])
                
                if not (     (LP is None) and  (LPn is None) 
                        and  (LM is None) and  (LMn is None) ):
                    LP  = np.dstack([LP[:,:,:1]  ,LP])
                    LPn = np.dstack([LPn[:,:,:1] ,LPn])
                    LM  = np.dstack([LM[:,:,:1]  ,LM])
                    LMn = np.dstack([LMn[:,:,:1] ,LMn])
            
            # Case2: x3[i] coincides with an element of X3vec
            elif L[-1] == x3[i]:
                X3vec  = X3vec
                Avec   = Avec
                B11vec = B11vec
                B13vec = B13vec
                B33vec = B33vec
                G1vec  = G1vec
                G3vec  = G3vec
                K3     = K3
                K3n    = K3n
                
                if not (     (LP is None) and  (LPn is None) 
                        and  (LM is None) and  (LMn is None) ):
                    LP  = LP
                    LPn = LPn
                    LM  = LM
                    LMn = LMn
            
            # Case 3: x3[i] is larger than X3vec[-1]
            elif L.size == X3vec.size:
                X3vec  = np.hstack([X3vec,x3[i]])
                Avec   = np.hstack([Avec  ,Avec[-1]])
                B11vec = np.hstack([B11vec,B11vec[-1]])
                B13vec = np.hstack([B13vec,B13vec[-1]])
                B33vec = np.hstack([B33vec,B33vec[-1]])
                G1vec  = np.hstack([G1vec ,G1vec[-1]])
                G3vec  = np.hstack([G3vec ,G3vec[-1]])
                K3     = np.dstack([K3    ,K3[:,:,-1:]])
                K3n    = np.dstack([K3n   ,K3n[:,:,-1:]])
                
                if not (     (LP is None) and  (LPn is None) 
                        and  (LM is None) and  (LMn is None) ):
                    LP  = np.dstack([LP  , LP[:,:,-1:]])
                    LPn = np.dstack([LPn , LPn[:,:,-1:]])
                    LM  = np.dstack([LM  , LM[:,:,-1:]])
                    LMn = np.dstack([LMn , LMn[:,:,-1:]])
                
            # Case 4: x3[i] is between X3vec[0] and X3vec[-1] AND does not 
            # coincide with any element of X3vec
            else:
                
                b = L[-1] 
                ind = X3vec.tolist().index(b)
                
                X3vec  = np.hstack([X3vec[:ind+1]  ,x3[i]               ,X3vec[ind+1:]])
                Avec   = np.hstack([Avec[:ind+1]   ,Avec[ind+1]         ,Avec[ind+1:]])
                B11vec = np.hstack([B11vec[:ind+1] ,B11vec[ind+1]       ,B11vec[ind+1:]])
                B13vec = np.hstack([B13vec[:ind+1] ,B13vec[ind+1]       ,B13vec[ind+1:]])
                B33vec = np.hstack([B33vec[:ind+1] ,B33vec[ind+1]       ,B33vec[ind+1:]])
                G1vec  = np.hstack([G1vec[:ind+1]  ,G1vec[ind+1]        ,G1vec[ind+1:]])
                G3vec  = np.hstack([G3vec[:ind+1]  ,G3vec[ind+1]        ,G3vec[ind+1:]])
                K3     = np.dstack([K3[:,:,:ind+1] ,K3[:,:,ind+1:ind+2] ,K3[:,:,ind+1:]])
                K3n    = np.dstack([K3n[:,:,:ind+1],K3n[:,:,ind+1:ind+2],K3n[:,:,ind+1:]])
                
                if not (     (LP is None) and  (LPn is None) 
                        and  (LM is None) and  (LMn is None) ):
                    LP  = np.dstack([LP[:,:,:ind+1] ,LP[:,:,ind+1:ind+2] ,LP[:,:,ind+1:]])
                    LPn = np.dstack([LPn[:,:,:ind+1],LPn[:,:,ind+1:ind+2],LPn[:,:,ind+1:]])
                    LM  = np.dstack([LM[:,:,:ind+1] ,LM[:,:,ind+1:ind+2] ,LM[:,:,ind+1:]])
                    LMn = np.dstack([LMn[:,:,:ind+1],LMn[:,:,ind+1:ind+2],LMn[:,:,ind+1:]])
            
        # Update self: Apply layer insertion to the self-parameters    
        if UpdateSelf is True:
            self.x3vec  = X3vec
            self.avec   = Avec
            self.b11vec = B11vec
            self.b13vec = B13vec
            self.b33vec = B33vec
            self.g1vec  = G1vec
            self.g3vec  = G3vec
            self.K3     = K3
            self.K3n    = K3n
            self.LP  = LP
            self.LPn = LPn
            self.LM  = LM
            self.LMn = LMn
            
            
        out = {'x3vec':X3vec,'avec':Avec,'b11vec':B11vec,'b13vec':B13vec,
               'b33vec':B33vec,'g1vec':G1vec,'g3vec':G3vec,'K3':K3,'K3n':K3n,
               'LP':LP,'LPn':LPn,'LM':LM,'LMn':LMn}
        
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
        >>> F = LM(nt=1024,dt=0.005,nr=4096,dx1=12.5,eps=3/(513*0.005),
                   x3vec=np.array([1.1,2.2,3.7])*1e3,
                   avec=np.array([1,2,3])*1e-3,b11vec=np.array([1.4,3.14,2])*1e-4,
                   b13vec=np.array([0.4,2.4,1.2])*1e-4,
                   b33vec=np.array([1.4,3.14,2])*1e-4
                   g1vec=np.array([0.8,2,1.3])*1e-4,
                   g3vec=np.array([1.8,0.7,2.3])*1e-4,
                   ReciprocalMedium=False,AdjointMedium=True) 
        
        >>> # Compute response to a source at x3=500 measured at x3=2000
        >>> G = Fe.GreensFunction_k1_w(x3S=500,x3R=2000,normalisation='flux',
        >>>                            InternalMultiples=True)
        >>> GPP = G['GPP']
        >>> GPM = G['GPM']
        >>> GMP = G['GMP']
        >>> GMM = G['GMM']
        
        >>> # Ricker wavelet
        >>> Wav = Fe.RickerWavelet_w(f0=30)
        
        >>> # Correct for complex-valued frequency
        >>> gain = Fe.Gain_t()

        >>> # Tranform to the space-time domain
        >>> gPP = np.fft.fftshift(gain*Fe.K1W2X1T(Wav*GPP),axes=(0,1))
        >>> gPM = np.fft.fftshift(gain*Fe.K1W2X1T(Wav*GPM),axes=(0,1))
        >>> gMP = np.fft.fftshift(gain*Fe.K1W2X1T(Wav*GMP),axes=(0,1))
        >>> gMM = np.fft.fftshift(gain*Fe.K1W2X1T(Wav*GMM),axes=(0,1))
        >>> # The resulting plot is shown below.
        
        .. image:: ../pictures/cropped/G_Functions.png
           :width: 300px
           :height: 200px
        
        
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
            
        X3vec  = Tmp_medium['x3vec']
        Avec   = Tmp_medium['avec']
        B11vec = Tmp_medium['b11vec']
        B13vec = Tmp_medium['b13vec']
        B33vec = Tmp_medium['b33vec']
        G1vec  = Tmp_medium['g1vec']
        G3vec  = Tmp_medium['g3vec']
        
        # Get indices of the receiver and source interfaces
        r = X3vec.tolist().index(x3R)
        s = X3vec.tolist().index(x3S)
        
        if x3R > x3S:
            
            # Overburden
            x3vec  = X3vec[:s+2]
            avec   = Avec[:s+2]
            b11vec = B11vec[:s+2]
            b13vec = B13vec[:s+2]
            b33vec = B33vec[:s+2]
            g1vec  = G1vec[:s+2]
            g3vec  = G3vec[:s+2]
            
            L1 = self.RT_response_k1_w(x3vec=x3vec,avec=avec,b11vec=b11vec,
                                      b13vec=b13vec,b33vec=b33vec,g1vec=g1vec,
                                      g3vec=g3vec,normalisation=normalisation,
                                      InternalMultiples=InternalMultiples)

            # Sandwiched layer stack
            x3vec  = X3vec[s+1:r+2] - X3vec[s]
            avec   = Avec[s+1:r+2]
            b11vec = B11vec[s+1:r+2]
            b13vec = B13vec[s+1:r+2]
            b33vec = B33vec[s+1:r+2]
            g1vec  = G1vec[s+1:r+2]
            g3vec  = G3vec[s+1:r+2]
            
            L2 = self.RT_response_k1_w(x3vec=x3vec,avec=avec,b11vec=b11vec,
                                      b13vec=b13vec,b33vec=b33vec,g1vec=g1vec,
                                      g3vec=g3vec,normalisation=normalisation,
                                      InternalMultiples=InternalMultiples)
            
            # Underburden
            x3vec  = X3vec[r+1:] - X3vec[r]
            avec   = Avec[r+1:]
            b11vec = B11vec[r+1:]
            b13vec = B13vec[r+1:]
            b33vec = B33vec[r+1:]
            g1vec  = G1vec[r+1:]
            g3vec  = G3vec[r+1:]
            
            # Exception if underburden is homogeneous
            if x3vec.size == 0:
                x3vec  = np.array([0])
                avec   = Avec[r:]
                b11vec = B11vec[r:]
                b13vec = B13vec[r:]
                b33vec = B33vec[r:]
                g1vec  = G1vec[r:]
                g3vec  = G3vec[r:]
            
            L3 = self.RT_response_k1_w(x3vec=x3vec,avec=avec,b11vec=b11vec,
                                      b13vec=b13vec,b33vec=b33vec,g1vec=g1vec,
                                      g3vec=g3vec,normalisation=normalisation,
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
            x3vec  = X3vec[:s+2]
            avec   = Avec[:s+2]
            b11vec = B11vec[:s+2]
            b13vec = B13vec[:s+2]
            b33vec = B33vec[:s+2]
            g1vec  = G1vec[:s+2]
            g3vec  = G3vec[:s+2]
            
            L1 = self.RT_response_k1_w(x3vec=x3vec,avec=avec,b11vec=b11vec,
                                      b13vec=b13vec,b33vec=b33vec,g1vec=g1vec,
                                      g3vec=g3vec,normalisation=normalisation,
                                      InternalMultiples=InternalMultiples)
            
            # Underburden
            x3vec  = X3vec[r+1:] - X3vec[r]
            avec   = Avec[r+1:]
            b11vec = B11vec[r+1:]
            b13vec = B13vec[r+1:]
            b33vec = B33vec[r+1:]
            g1vec  = G1vec[r+1:]
            g3vec  = G3vec[r+1:]
            
            # Exception if underburden is homogeneous
            if x3vec.size == 0:
                x3vec = np.array([0])
                avec  = Avec[r:]
                b11vec = B11vec[r:]
                b13vec = B13vec[r:]
                b33vec = B33vec[r:]
                g1vec = G1vec[r:]
                g3vec = G3vec[r:]
            
            L3 = self.RT_response_k1_w(x3vec=x3vec,avec=avec,b11vec=b11vec,
                                       b13vec=b13vec,b33vec=b33vec,g1vec=g1vec,
                                       g3vec=g3vec,normalisation=normalisation,
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
            x3vec  = X3vec[:r+2]
            avec   = Avec[:r+2]
            b11vec = B11vec[:r+2]
            b13vec = B13vec[:r+2]
            b33vec = B33vec[:r+2]
            g1vec  = G1vec[:r+2]
            g3vec  = G3vec[:r+2]
            
            L1 = self.RT_response_k1_w(x3vec=x3vec,avec=avec,b11vec=b11vec,
                                       b13vec=b13vec,b33vec=b33vec,g1vec=g1vec,
                                       g3vec=g3vec,normalisation=normalisation,
                                       InternalMultiples=InternalMultiples)
            
            # Sandwiched layer stack
            x3vec  = X3vec[r+1:s+2] - X3vec[r]
            avec   = Avec[r+1:s+2]
            b11vec = B11vec[r+1:s+2]
            b13vec = B13vec[r+1:s+2]
            b33vec = B33vec[r+1:s+2]
            g1vec  = G1vec[r+1:s+2]
            g3vec  = G3vec[r+1:s+2]
            
            L2 = self.RT_response_k1_w(x3vec=x3vec,avec=avec,b11vec=b11vec,
                                       b13vec=b13vec,b33vec=b33vec,g1vec=g1vec,
                                       g3vec=g3vec,normalisation=normalisation,
                                       InternalMultiples=InternalMultiples)
            
            # Underburden
            x3vec  = X3vec[s+1:] - X3vec[s]
            avec   = Avec[s+1:]
            b11vec = B11vec[s+1:]
            b13vec = B13vec[s+1:]
            b33vec = B33vec[s+1:]
            g1vec  = G1vec[s+1:]
            g3vec  = G3vec[s+1:]
            
            # Exception if underburden is homogeneous
            if x3vec.size == 0:
                x3vec  = np.array([0])
                avec   = Avec[s:]
                b11vec = B11vec[s:]
                b13vec = B13vec[s:]
                b33vec = B33vec[s:]
                g1vec  = G1vec[s:]
                g3vec  = G3vec[s:]
            
            L3 = self.RT_response_k1_w(x3vec=x3vec,avec=avec,b11vec=b11vec,
                                       b13vec=b13vec,b33vec=b33vec,g1vec=g1vec,
                                       g3vec=g3vec,normalisation=normalisation,
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
    
    def FocusingFunction_k1_w(self,x3F,normalisation='flux',
                              InternalMultiples=True,Negative_eps=False,
                              UpdateSelf=False):
        """computes the focusing functions between the top surface 
        (:math:`x_3=0`) and the focusing depth defined by the input variable 
        \'x3F\'. 
        
        We define the focusing depth just below \'x3F\'. Hence, if the 
        focusing depth coincides with an interface the focusing function 
        focuses below that interface.
        
        Parameters
        ----------
    
        x3F : int,float
            Focusing depth.
            
        normalisation : str, optional
            For pressure-normalisation set normalisation='pressure', for 
            flux-normalisation set normalisation='flux'. Until now, this 
            function only models the focusing function for flux-normalisation.
            For pressure-normalisation, I still have to derive the analytic 
            expression to recursively compute the focusing function.
            
        InternalMultiples : bool, optional
            To model internal multiples set 'InternalMultiples=True'. To ignore 
            internal multiples set 'InternalMultiples=False'.
            
        Negative_eps : bool, optional
            If 'Negative_eps=True' the focusing functions are computed for both 
            a positive and a negative parameter 'eps' (imaginary-part of the
            frequency,
            :math:`\omega\' = \omega + j \epsilon`). 
            The representation theorem of the corrleation-type only holds 
            accurately if the complex-conjugated focusing functions are 
            computed with a negative 'eps' parameter, whereas all other 
            qunatities (Green's functions and reflection response) must be 
            computed using a positive 'eps' parameter.
            
        UpdateSelf : bool, optional
            If 'UpdateSelf=True' the focusing depth level 'x3F' is inserted in 
            the model.
            
            
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
                - **FP_neps**: Downgoing focusing function with negative 'eps' parameter, :math:`\omega\' = \omega - j \\vert \epsilon \\vert`.
                - **FM_neps**: Upgoing focusing function with negative 'eps' parameter, :math:`\omega\' = \omega - j \\vert \epsilon \\vert`.
                - **FPa_neps**: Downgoing focusing function with negative 'eps' parameter, :math:`\omega\' = \omega - j \\vert \epsilon \\vert`,  in adjoint medium.
                - **FMa_neps**: Upgoing focusing function with negative 'eps' parameter, :math:`\omega\' = \omega - j  \\vert \epsilon \\vert`,  in adjoint medium.
                - **RPa_neps**: Reflection response from above  with negative 'eps' parameter, :math:`\omega\' = \omega - j \\vert \epsilon \\vert`,  in adjoint medium.
                
            All medium responses are stored in arrays of shape (nf,nr). The 
            variables 'FPa', 'RPa', 'TPa', 'FMa', 'RMa' and 'TMa' are computed 
            only if one sets 'AdjointMedium=True'. The variables ending on 
            '_neps' are only computed if one set 'Negative_eps=True'.
        
        
        Notes
        -----
        
        - The downgoing focusing funtion :math:`\\tilde{F}_1^+` is computed by inverting the expressions for the transmission from above :math:`\\tilde{T}^+`:
            :math:`\\tilde{F}_{1,n}^+ = \\tilde{F}_{1,n-1}^+ (\\tilde{w}_n^+)^{-1} (1 - \\tilde{w}_n^+ \\tilde{R}_{n-1}^{\cap} \\tilde{w}_n^- \\tilde{r}_n^{\cup} )^{-1} (\\tilde{t}_n^+)^{-1}`
        - The upgoing focusing function is computed by applying the reflection response :math:`R^{\cup}` on the downgoing focusing funtion :math:`\\tilde{F}_1^+`:
            :math:`\\tilde{F}_{1,n}^- = \\tilde{R}^{\cup} \\tilde{F}_{1,n}^+`.
        - When using a complex-valued frequency :math:`\omega' = \omega + j \epsilon`, the representation theorem of the corrleation-type only holds accurately if the complex-conjugated focusing functions are computed with a negative 'eps' parameter, whereas all other qunatities (Green's functions and reflection response) must be computed using a positive 'eps' parameter.
        
        
        References
        ----------
        Kees document as soon as it is published.
        
        
        Examples
        --------

        >>> from Layered_NRM_k1_w import Layered_NRM_k1_w as LM
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

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
        
        >>> # Compute Ricker wavelet
        >>> Wav = F.RickerWavelet_w(f0=30)
        
        >>> # Compute correction term
        >>> gain = F.Gain_t()
        
        >>> # Make FK-filter
        >>> M = F.FK1_mask_k1_w(wmax=400,Opening=1.0)
        >>> FK = M['FK_global_tap']
        
        >>> # Focusing depth
        >>> xF = 2200
        
        >>> # Compute focusing function in actual medium
        >>> Focus = F.FocusingFunction_k1_w(x3F=xF,normalisation='flux',
                                            InternalMultiples=True,
                                            Negative_eps=True)
        >>> FP = Focus['FP_neps']
        >>> FM = Focus['FM_neps']
        
        >>> # Compute reflection response of actual medium
        >>> RT = F.RT_response_k1_w(normalisation='flux',InternalMultiples=True)
        >>> RP = RT['RP']
        
        >>> # Compute Green's function G++ in actual medium
        >>> G = F.GreensFunction_k1_w(x3S=0,x3R=xF,normalisation='flux',
        >>>                           InternalMultiples=True)
        >>> GPP = G['GPP']
        
        >>> # Transform all fields to the space-time domain
        >>> gPP   = np.fft.fftshift(gain*F.K1W2X1T(FK*Wav*GPP)          ,axes=(0,1))
        >>> fPc   = np.fft.fftshift(gain*F.K1W2X1T(FK*Wav*FP.conj())    ,axes=(0,1))
        >>> rPfMc = np.fft.fftshift(gain*F.K1W2X1T(FK*Wav*RP*FM.conj()) ,axes=(0,1))
        >>> # We plot the resulting fields below.

        
        .. image:: ../pictures/cropped/RecipCorr.png
           :width: 700px
           :height: 200px
        
        """
        
        # Check if normalisation is set correctly
        if normalisation is not 'flux':
            sys.exit('FocusingFunction_k1_w: This function only models the '
                     +'focusing function for flux-normalisation. (For '
                     +'pressure-normalistiont the required equations have to '
                     +'be derived.)')
            
        # Check if UpdateSelf is a bool    
        if not isinstance(UpdateSelf,bool):
            sys.exit('FocusingFunction_k1_w: The input variable \'UpdateSelf\''
                     +' must be a bool.')
            
        # Eigenvalues: To ensure that the eigenvalues are defined as 'self'
        # parameters
        Eig = self.Eigenvalues_k1_w()
        
        # Insert transparent interfaces at the focusing depth level.
        # The insertion implicitly checks that x3F is non-negative 
        # and real-valued.
        # If the focusing depth is greater than, or equal to the deepest 
        # interface, we insert another transparent layer below the focusing 
        # depth to be able to compute scattering coefficients at the focusing 
        # depth without getting an index error.
        if x3F >= self.x3vec[-1]:
            Tmp_medium = self.Insert_layer(x3=np.array([x3F,x3F+1]),
                                           UpdateSelf=UpdateSelf)
        else:
            Tmp_medium = self.Insert_layer(x3=x3F,UpdateSelf=UpdateSelf)
        
        
        
        X3  = Tmp_medium['x3vec']
        A   = Tmp_medium['avec']
        B11 = Tmp_medium['b11vec']
        B13 = Tmp_medium['b13vec']
        B33 = Tmp_medium['b33vec']
        G1  = Tmp_medium['g1vec']
        G3  = Tmp_medium['g3vec']
        K3  = Tmp_medium['K3']
        K3n = Tmp_medium['K3n']
        LP  = Tmp_medium['LP']
        LPn = Tmp_medium['LPn']
        LM  = Tmp_medium['LM']
        LMn = Tmp_medium['LMn']
        
        # Index of the focusing depth
        f = X3.tolist().index(x3F)
        
        # Vector with layer thicknesses
        dx3vec = X3.copy()
        dx3vec[1:] = X3[1:]-X3[:-1]
        
        # Down- and upgoing focusing functions: Initial value
        # Here every frequency component has an amplitude equal to one. Hence,
        # the total wavefield has a strength of sqrt(nt)
        # When an ifft is applied the wavefield is scaled by 1/sqrt(nt).
        # Hence in the time domain the wavefield has an amplitude equal to one.
        FP = np.ones((self.nf,self.nr),dtype=complex)
        FM = np.zeros((self.nf,self.nr),dtype=complex)
        
        # Reflection responses: Initial value
        RP = np.zeros((self.nf,self.nr),dtype=complex)
        RM = np.zeros((self.nf,self.nr),dtype=complex)
        
        # Here every frequency component has an amplitude equal to one. Hence,
        # the total wavefield has a strength of sqrt(nt)
        # When an ifft is applied the wavefield is scaled by 1/sqrt(nt).
        # Hence in the time domain the wavefield has an amplitude equal to one.
        TP = np.ones((self.nf,self.nr),dtype=complex)
        TM = np.ones((self.nf,self.nr),dtype=complex)
        
        # Internal multiple operator: Initial value
        M1 = np.ones((self.nf,self.nr),dtype=complex)
        M2 = np.ones((self.nf,self.nr),dtype=complex)
        M3 = np.ones((self.nf,self.nr),dtype=complex)
        
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
            ScatCoeffs = self.RT_k1_w(beta11_u=B11[n],beta13_u=B13[n],
                                      beta33_u=B33[n],
                                      K3_u=K3[:,:,n],K3n_u=K3n[:,:,n],
                                      beta11_l=B11[n+1],beta13_l=B13[n+1],
                                      beta33_l=B33[n+1],
                                      K3_l=K3[:,:,n+1],K3n_l=K3n[:,:,n+1],
                                      normalisation=normalisation)
            
            rP = ScatCoeffs['rP']
            tP = ScatCoeffs['tP']
            rM = ScatCoeffs['rM']
            tM = ScatCoeffs['tM']
            
            # Propagators
            if self.AdjointMedium is True:
                W = self.W_propagators_k1_w(LP=LP[:,:,n],LPn=LPn[:,:,n],
                                            LM=LM[:,:,n],LMn=LMn[:,:,n],
                                            dx3=dx3vec[n])
            else:
                W = self.W_propagators_k1_w(LP=LP[:,:,n],LM=LM[:,:,n],
                                            dx3=dx3vec[n])
                
            WP  = W['wP']
            WM  = W['wM']
        
            if InternalMultiples is True:
                M1 = 1 / (1 - RM*WM*rP*WP)
                M2 = 1 / (1 - rP*WP*RM*WM)
                M3 = (1 - RM*WM*rP*WP)
            
            # Update focusing functions and reflection / transmission responses
            FP = FP*M3/WP/tP
            RP = RP + TM*WM*rP*WP*M1*TP
            RM = rM + tP*WP*RM*WM*M2*tM
            TP = tP*WP*M1*TP
            TM = TM*WM*M2*tM  
            FM = RP*FP
            
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
                    M3 = (1 - RMa*WM*rP*WP)      
            
                # Model focusing functions in the adjoint medium
                # Update focusing functions and reflection / transmission responses
                FPa = FPa*M3/WP/tP
                RPa = RPa + TMa*WM*rP*WP*M1*TPa
                RMa = rM + tP*WP*RMa*WM*M2*tM
                TPa = tP*WP*M1*TPa
                TMa = TMa*WM*M2*tM  
                FMa = RPa*FPa
                
        # Verbose: Inform the user if any wavefield contains NaNs of Infs.
        if self.verbose is True:
            self.Contains_Nan_Inf('FocusingFunction_k1_w',(FP,'FP'),(FM,'FM'),
                                  (RP,'RP'),(RM,'RM'),(TP,'TP'),(TM,'TM'))
            
            if self.AdjointMedium is True:
                self.Contains_Nan_Inf('FocusingFunction_k1_w',(FPa,'FPa'),
                                      (FMa,'FMa'),(RPa,'RPa'),(RMa,'RMa'),
                                      (TPa,'TPa'),(TMa,'TMa'))
                
        out={'FP':FP  ,'RP':RP  ,'TP':TP  ,'FM':FM  ,'RM':RM  ,'TM':TM,
                 'FPa':FPa,'RPa':RPa,'TPa':TPa,'FMa':FMa,'RMa':RMa,'TMa':TMa}
        
        # When focusing functions are complex-conjugated the imaginary-part
        # of a complex-valued frequency, eps, should be negative.
        # If requested the corresponding focusing functions are computed below.
        if (self.eps is not None) and (self.eps != 0) and (Negative_eps is True):
            self.SubSelf = Layered_NRM_k1_w(self.nt,self.dt,self.nr,self.dx1,
                                            self.verbose,eps=-self.eps,
                                            x3vec=X3,avec=A,
                                            b11vec=B11,b13vec=B13,b33vec=B33,
                                            g1vec=G1,g3vec=G3,
                                            ReciprocalMedium=self.ReciprocalMedium,
                                            AdjointMedium=self.AdjointMedium)
                
            # Eigenvalues: 
            Eig = self.SubSelf.Eigenvalues_k1_w()
            LP  = Eig['LP']
            LM  = Eig['LM']
            LPn = Eig['LPn']
            LMn = Eig['LMn']
            
            # Vertical wavenumbers
            K3  = self.SubSelf.K3
            K3n = self.SubSelf.K3n
            
            # Down- and upgoing focusing functions: Initial value
            # Here every frequency component has an amplitude equal to one. Hence,
            # the total wavefield has a strength of sqrt(nt)
            # When an ifft is applied the wavefield is scaled by 1/sqrt(nt).
            # Hence in the time domain the wavefield has an amplitude equal to one.
            FP = np.ones((self.nf,self.nr),dtype=complex)
            FM = np.zeros((self.nf,self.nr),dtype=complex)
            
            # Reflection responses: Initial value
            RP = np.zeros((self.nf,self.nr),dtype=complex)
            RM = np.zeros((self.nf,self.nr),dtype=complex)
            
            # Here every frequency component has an amplitude equal to one. Hence,
            # the total wavefield has a strength of sqrt(nt)
            # When an ifft is applied the wavefield is scaled by 1/sqrt(nt).
            # Hence in the time domain the wavefield has an amplitude equal to one.
            TP = np.ones((self.nf,self.nr),dtype=complex)
            TM = np.ones((self.nf,self.nr),dtype=complex)
            
            # Internal multiple operator: Initial value
            M1 = np.ones((self.nf,self.nr),dtype=complex)
            M2 = np.ones((self.nf,self.nr),dtype=complex)
            
            # Adjoint medium
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
                ScatCoeffs = self.SubSelf.RT_k1_w(beta11_u=B11[n],beta13_u=B13[n],
                                                  beta33_u=B33[n],
                                                  K3_u=K3[:,:,n],K3n_u=K3n[:,:,n],
                                                  beta11_l=B11[n+1],beta13_l=B13[n+1],
                                                  beta33_l=B33[n+1],
                                                  K3_l=K3[:,:,n+1],K3n_l=K3n[:,:,n+1],
                                                  normalisation=normalisation)
                rP = ScatCoeffs['rP']
                tP = ScatCoeffs['tP']
                rM = ScatCoeffs['rM']
                tM = ScatCoeffs['tM']
                
                # Propagators
                if self.AdjointMedium is True:
                    W = self.SubSelf.W_propagators_k1_w(LP=LP[:,:,n],LPn=LPn[:,:,n],
                                                        LM=LM[:,:,n],LMn=LMn[:,:,n],
                                                        dx3=dx3vec[n])
                else:
                    W = self.SubSelf.W_propagators_k1_w(LP=LP[:,:,n],LM=LM[:,:,n],
                                                        dx3=dx3vec[n])
                    
                WP  = W['wP']
                WM  = W['wM']
            
                if InternalMultiples is True:
                    M1 = 1 / (1 - RM*WM*rP*WP)
                    M2 = 1 / (1 - rP*WP*RM*WM)
                    M3 = (1 - RM*WM*rP*WP)
                
                # Update focusing functions and reflection / transmission responses
                FP = FP*M3/WP/tP
                RP = RP + TM*WM*rP*WP*M1*TP
                RM = rM + tP*WP*RM*WM*M2*tM
                TP = tP*WP*M1*TP
                TM = TM*WM*M2*tM  
                FM = RP*FP
                
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
                        M3 = (1 - RMa*WM*rP*WP)  
                    
                    # Model focusing functions in the adjoint medium
                    # Update focusing functions and reflection / transmission responses
                    FPa = FPa*M3/WP/tP
                    RPa = RPa + TMa*WM*rP*WP*M1*TPa
                    RMa = rM + tP*WP*RMa*WM*M2*tM
                    TPa = tP*WP*M1*TPa
                    TMa = TMa*WM*M2*tM  
                    FMa = RPa*FPa
                    
            # Verbose: Inform the user if any wavefield contains NaNs of Infs.
            if self.verbose is True:
                self.Contains_Nan_Inf('FocusingFunction_k1_w',(FP,'FP'),(FM,'FM'),
                                      (RP,'RP'),(RM,'RM'),(TP,'TP'),(TM,'TM'))
                
                if self.AdjointMedium is True:
                    self.Contains_Nan_Inf('FocusingFunction_k1_w',(FPa,'FPa'),
                                          (FMa,'FMa'),(RPa,'RPa'),(RMa,'RMa'),
                                          (TPa,'TPa'),(TMa,'TMa'))
                    
            out.update({'FP_neps':FP   ,'FM_neps':FM  ,'RP_neps':RP ,
                        'FPa_neps':FPa ,'FMa_neps':FMa})
        return out
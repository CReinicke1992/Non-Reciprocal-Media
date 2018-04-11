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

from Wavefield_NRM_p_w import Wavefield_NRM_p_w
import numpy as np
import sys

class Layered_NRM_p_w(Wavefield_NRM_p_w):
    """is a class to model wavefields in 1.5D (non-)reciprocal media in the ray-parameter frequency domain.
        
    The class Layered_NRM_p_w defines a 1.5D (non-)reciprocal medium and a scalar wavefield. We consider a single horizontal ray-parameter 'p1' and all frequencies that are sampled by the given number of time samples 'nt' and the time sample interval 'dt'.

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
        
    x3vec : numpy.ndarray
        Vertical spatial vector :math:`x_3`, for n layers 'x3vec' must have the shape (n,). We define the :math:`x_3`-axis as downward-pointing. Implicitly, the first value on the :math:`x_3`-axis is zero (not stored in 'x3vec').
    
    avec : numpy.ndarray
        Medium parameter :math:`\\alpha` (real-valued), for n layers 'avec' must have the shape (n,).
        
    bvec : numpy.ndarray
        Medium parameter :math:`\\beta` (real-valued), for n layers 'bvec' must have the shape (n,).
    
    g1vec : numpy.ndarray, optional
        Medium parameter :math:`\gamma_1` (real-valued for non-reciprocal media or imaginary-valued for reciprocal media), for n layers 'g1vec' must have the shape (n,).
        
    g3vec : numpy.ndarray, optional
        Medium parameter :math:`\gamma_3` (real-valued for non-reciprocal media or imaginary-valued for reciprocal media), for n layers 'g3vec' must have the shape (n,).
        
    p1 : int, float
        Horizontal ray-parameter in seconds per metre.
        
    ReciprocalMedium : bool, optional
        For non-reciprocal media set 'ReciprocalMedium=False', for reciprocal media set 'ReciprocalMedium=True'.
        
    AdjointMedium : bool, optional
        Set 'AdjointMedium=True' to compute scattering coefficients and propagators in an adjoint medium :math:`^{(a)}`. For reciprocal media, the scattering coefficients and propagators are identical in a medium and its adjoint. We have defined the scattering and propagation in the adjoint medium only for flux-normalisation.
        
    Returns
    -------
    
    class
        A class to model a wavefield in a 1.5D non-reciprocal medium in the ray-parameter frequency domain. The following instances are defined:
             - **x3vec**: :math:`x_3`.
            - **avec**: :math:`\\alpha`.
            - **bvec**: :math:`\\beta`.
            - **g1vec**: :math:`\gamma_1`.
            - **g3vec**: :math:`\gamma_3`.
            - **p1**: Horizontal ray-parameter.
            - **ReciprocalMedium**: True for reciprocal media, False for non-reciprocal media.
            - **AdjointMedium**: If True, propagation and scatteing are defined in a medium and in its adjoint.
            - **p3**: Vertical ray-parameter for positive 'p1'.
            - **p3n**: Vertical ray-parameter for negative 'p1'.
        
    Notes
    -----
    - We format the data as described below.
        - Wavefields are saved in an array of dimensions (nf,nr) in the frequency domain and (nt,nr) in the time domain.
        - Wavefields are in the p- :math:`\omega` domain.
        - The zero frequency component is placed at the first index position.
        - If the wavefield is transformed to the time domain, the zero time component is placed at the first index position, followed by nt/2-1 positive time samples and nt/2 negative time samples.
    - For evanescent waves, Kees makes a sign choice for the vertical ray-parameter,
        - :math:`p_3' = -j \sqrt{p_1^2 - (\\alpha \\beta + \gamma_1^2 + \gamma_3^2)}`.
      By default, **NumPy** makes the oppostie sign choice, 
          - :math:`p_3' = +j \sqrt{p_1^2 - (\\alpha \\beta + \gamma_1^2 + \gamma_3^2)}`.
      We stick to the sign choice by **NumPy**. Thus, we will also change the sign choice for the propagators,
          - Kees chose: :math:`\\tilde{w}^{\pm} = \mathrm{exp}(-j \omega p_3' \Delta x_3)`.
          - We choose: :math:`\\tilde{w}^{\pm} = \mathrm{exp}(+j \omega p_3' \Delta x_3)`.
        
    References
    ----------
    Kees document as soon as it is published.
     
    Examples
    --------

    >>> from Layered_NRM_p_w import Layered_NRM_p_w as LM
    >>> import numpy as np
    
    >>> # Initialise wavefield in a layered non-reciprocal medium
    >>> F=LM(nt=1024,dt=0.005,x3vec=np.array([1.1,2.2,3.7]),avec=np.array([1,2,3]),bvec=np.array([0.4,3.14,2]),p1=2e-4,ReciprocalMedium=False)
        
    .. csv-table:: **Unified scalar wavefields: Quantities**
        "**Wave- field**", "**P**", ":math:`\mathbf{Q}_1`", ":math:`\mathbf{Q}_3`", ":math:`\mathbf{\\alpha}`", ":math:`\mathbf{\\beta}`", ":math:`\mathbf{\gamma}_1`", ":math:`\mathbf{\gamma}_3`", ":math:`\mathbf{\delta}_1`", ":math:`\mathbf{\delta}_3`", ":math:`\mathbf{B}`", ":math:`\mathbf{C}_1`", ":math:`\mathbf{C}_3`"
        "*TE*", ":math:`E_2`", ":math:`H_3`", ":math:`-H_1`", ":math:`\epsilon`", ":math:`\mu`", ":math:`\\xi_{23}`", ":math:`-\\xi_{21}`", ":math:`\zeta_{32}`", ":math:`-\zeta_{12}`", ":math:`-J_2^e`", ":math:`-J_3^m`", ":math:`J_1^m`"
        "*TM*", ":math:`H_2`", ":math:`-E_3`", ":math:`E_1`", ":math:`\mu`", ":math:`\epsilon`", ":math:`-\zeta_{23}`", ":math:`\zeta_{21}`", ":math:`-\\xi_{32}`", ":math:`\\xi_{12}`", ":math:`-J_2^m`", ":math:`J_3^e`", ":math:`-J_1^e`"
       "*Ac. (fluid)*", ":math:`p`", ":math:`v_1`", ":math:`v_3`", ":math:`\kappa`", ":math:`\\rho`", ":math:`d_1`", ":math:`d_3`", ":math:`e_1`", ":math:`e_3`", ":math:`q`", ":math:`f_1`", ":math:`f_3`"
       "*SH (solid)*", ":math:`v_2`", ":math:`-\\tau_{21}`", ":math:`-\\tau_{23}`", ":math:`\\rho`", ":math:`1/\mu`", ":math:`e_1`", ":math:`e_3`", ":math:`d_1`", ":math:`d_3`", ":math:`f_2`", ":math:`2h_{21}`", ":math:`2h_{23}`" 
    
   """    
  
    def __init__(self,nt,dt,nr=1,dx1=1,verbose=False,x3vec=np.zeros(1),avec=np.zeros(1),bvec=np.zeros(1),g1vec=np.zeros(1),g3vec=np.zeros(1),p1=None,ReciprocalMedium=False,AdjointMedium=False):
        
        # Inherit __init__ from Wavefield_NRM_p_w
        Wavefield_NRM_p_w.__init__(self,nt,dt,nr,dx1,verbose)
        
        # Check if medium parameters are passed as arrays
        if not ( isinstance(x3vec,np.ndarray) and isinstance(avec,np.ndarray) and isinstance(bvec,np.ndarray) and isinstance(g1vec,np.ndarray) and isinstance(g3vec,np.ndarray) ):
            sys.exit('Layered_NRM_p_w: x3vec, avec, bvec, g1vec and g3vec have to be of the type numpy.ndarray.')
            
        # Set gamma_1 and gamma_3 by default equal to zero
        if g1vec.all() == np.zeros(1):
            g1vec = np.zeros_like(avec)
        if g3vec.all() == np.zeros(1):
            g3vec = np.zeros_like(avec)
            
        # Force the medium parameters to have identical shape
        if x3vec.shape!=avec.shape or x3vec.shape!=bvec.shape or x3vec.shape!=g1vec.shape or x3vec.shape!=g3vec.shape:
            sys.exit('Layered_NRM_p_w: x3vec, avec, bvec, g1vec and g3vec have to be of identical shape.')
        
        # Force the medium parameters to be 1-dimensional, i.e. e.g. avec.shape=(n,)
        if x3vec.ndim!=1:
            sys.exit('Layered_NRM_p_w: x3vec.ndim, avec.ndim, bvec.ndim, g1vec.ndim and g3vec.ndim must be one.')
        
        # Check if x3vec is positive and constantly increasing
        if x3vec[0]<=0 or (x3vec[1:]-x3vec[:-1] <= 0).any():
            sys.exit('Layered_NRM_p_w: x3vec must only contain positive and constantly increasing values.')
        
        # Check if p1 is a scalar
        if not ( isinstance(p1,int) or isinstance(p1,float) ):
            sys.exit('Layered_NRM_p_w: p1 must be of tyoe int or float.')
        
        # Check if Medium choices are bools
        if not ( isinstance(ReciprocalMedium,bool) and isinstance(AdjointMedium,bool) ):
            sys.exit('Layered_NRM_p_w: \'ReciprocalMedium\' and \'AdjointMedium\' must be of the type bool.')
        
        # Check if medium parameters correspond to a lossless (non-)reciprocal medium
        if ReciprocalMedium == False:
            if avec.imag.any()!=0 or bvec.imag.any()!=0 or g1vec.imag.any()!=0 or g3vec.imag.any()!=0:
                sys.exit('Layered_NRM_p_w: In lossless non-reciprocal media the imaginary value of avec, bvec, g1vec and g3vec has to be zero.')
        elif ReciprocalMedium == True:
            if avec.imag.any()!=0 or bvec.imag.any()!=0 or g1vec.real.any()!=0 or g3vec.real.any()!=0:
                sys.exit('Layered_NRM_p_w: In lossless reciprocal media the imaginary value of avec and bvec has to be zero, the real value of g1vec and g3vec has to be zero.')
            
        # Set medium parameters 
        self.x3vec = x3vec
        self.avec = avec
        self.bvec = bvec
        self.g1vec = g1vec
        self.g3vec = g3vec
        self.p1 = p1
        self.ReciprocalMedium = ReciprocalMedium
        self.AdjointMedium = AdjointMedium
        
        # Calculate vertical ray-parameter p3=p3(+p1) p3n=p3(-p1) 
        # Note: By default python uses opposite sign convention for evanescent waves as Kees': (-1)**0.5=1j
        p3 = np.zeros(np.shape(self.avec),dtype=complex)
        p3n = np.zeros(np.shape(self.avec),dtype=complex)
        if self.ReciprocalMedium is True:
            p3[:] = self.avec*self.bvec + self.g1vec**2 + self.g3vec**2 - self.p1**2
            p3n[:] = p3.copy()
        elif self.ReciprocalMedium is False:
            p3[:] = self.avec*self.bvec - self.g1vec**2 + 2*self.g1vec*self.p1 - self.p1**2
            p3n[:] = self.avec*self.bvec - self.g1vec**2 - 2*self.g1vec*self.p1 - self.p1**2
        self.p3 = p3**0.5
        self.p3n = p3n**0.5
        
    def L_eigenvectors_p_w(self,beta=None,g3=None,p3=None,p3n=None,normalisation='flux'):
        """computes the eigenvector matrix 'L' and its inverse 'Linv', either in flux- or in pressure-normalisation for a single vertical ray-parameter 'p3' inside a homogeneous layer. If \'AdjointMedium=True\', **L_eigenvectors_p_w** also computes the eigenvector matrix in the adjoint medium 'La' and its inverse 'Lainv'.
        
        Parameters
        ----------
    
        beta : int, float
            Medium parameter :math:`\\beta`  (real-valued).
        
        g3 : int, float
            Medium parameter :math:`\gamma_3`.
        
        p3 : int, float
            Vertical ray-parameter :math:`p_3` for a positive horizontal ray-parameter :math:`p_1`.
        
        p3n : int, float, optional (required if 'AdjointMedium=True')
            Vertical ray-parameter :math:`p_3` for a negative horizontal ray-parameter :math:`p_1`.
            
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
            All eigenvector matrices are stored in a in a (2x2)-array. 
        
        Notes
        -----
            - The eigenvector matrix 'L' and its inverse 'Linv' are different for reciprocal and non-reciprocal media.
            - For reciprocal media, the eigenvectors of the adjoint medium are identical to the eigenvectors of the true medium.
            - We have defined the eigenvectors of the adjoint medium only for flux-normalisation.
        
        References
        ----------
        Kees document as soon as it is published.

        Examples
        --------

        >>> from Layered_NRM_p_w import Layered_NRM_p_w as LM
        >>> import numpy as np
        
        >>> # Initialise wavefield in a layered non-reciprocal medium
        >>> F=LM( nt=1024 , dt=0.005 , x3vec=np.array([1.1,2.2,3.7]) , avec=np.array([1,2,3]) , bvec=np.array([0.4,3.14,2]) , g1vec=np.array([0.9,2.1,0.3]) , g3vec=np.array([0.7,1.14,0.2]) , p1=2e-4 , ReciprocalMedium=False )
        
        >>> # Compute eigenvectors in flux-normalisation
        >>> Lvecs=F.L_eigenvectors_p_w(beta=0.1,g3=0.4,p3=2e-4,normalisation='flux')
        
        >>> # Eigenvector matrix
        >>> L = Lvecs['L']
        ([[15.8113883 +0.j, 15.8113883 +0.j],[ 0.03162278+0.j, -0.03162278+0.j]])
        
        >>> # Inverse eigenvector matrix
        >>> Linv = Lvecs['Linv']
        ([[  0.03162278+0.j,  15.8113883 +0.j],[  0.03162278+0.j, -15.8113883 -0.j]])
        
        """
        # Check if required input variables are given
        if (beta is None) or (g3 is None) or (p3 is None):
            sys.exit('L_eigenvectors_p_w: The input variables \'beta\', \'g3\' and  \'p3\' must be set.')
         
        # Check if normalisation is set correctly
        if (normalisation is not 'flux') and (normalisation is not 'pressure'):
            sys.exit('L_eigenvectors_p_w: The input variable \'normalisation\' must be set, either to \'flux\', or to \'pressure\'.')
            
        # Check if the vertical ray-parameter for a negative horizontal ray-parameter is given
        if  (self.AdjointMedium is True) and (p3n is None) and (normalisation is 'flux') and (self.ReciprocalMedium is False):
            sys.exit('L_eigenvectors_p_w: The input variable \'p3n\' (vertical ray-parameter p3 for a negative horizontal ray-parameter p1) must be given to compute the eigenvector matrix of the adjoint medium \'La\' and its inverse\'Lainv\'.')
            
        # Initialise L and Linv
        L = np.zeros((2,2),dtype=complex)
        Linv = np.zeros((2,2),dtype=complex)
        La = None
        Lainv = None
            
        if (self.ReciprocalMedium is True) and (normalisation is 'flux'):
            # L matrix
            L[0,0] = 1
            L[0,1] = 1
            L[1,0] = (p3+g3)/beta
            L[1,1] = -(p3-g3)/beta
            L = (beta/(2*p3))**0.5*L
            
            # Inverse L matrix
            Linv[0,0] = -L[1,1]
            Linv[0,1] = L[0,0]
            Linv[1,0] = L[1,0]
            Linv[1,1] = -L[0,0]
            
            if self.AdjointMedium is True:
                if self.verbose is True:
                    print('\nL_eigenvectors_p_w (AdjointMedium is True) and (ReciprocalMedium is True)')
                    print('---------------------------------------------------------------------------')
                    print('For reciprocal media, the eigenvector matrix of a medium and its adjoint medium are identical.\n')
               
                La = L.copy()
                Lainv = Linv.copy()
            
        elif (self.ReciprocalMedium is True) and (normalisation is 'pressure'):            
            # L matrix
            L[0,0] = 1
            L[0,1] = 1
            L[1,0] = (p3+g3)/beta
            L[1,1] = -(p3-g3)/beta
            
            # Inverse L matrix
            Linv[0,0] = -L[1,1]
            Linv[0,1] = 1
            Linv[1,0] = L[1,0]
            Linv[1,1] = -1
            Linv = beta/(2*p3)*Linv
            
            if self.verbose is True and self.AdjointMedium is True:
                print('\nL_eigenvectors_p_w (AdjointMedium is True) and (normalisation=\'pressure\')')
                print('-----------------------------------------------------------------------------')
                print('We have defined the eigenvector matrix of the adjoint medium \'La\' and its inverse \'Lainv\' only for flux-normalisation.\n')
            
        elif (self.ReciprocalMedium is False) and (normalisation is 'flux'):
            # L matrix
            L[0,0] = (beta/p3)**0.5
            L[0,1] = L[0,0]
            L[1,0] = (p3/beta)**0.5
            L[1,1] = -L[1,0]
            L = L/2**0.5
            
            # Inverse L matrix
            Linv[0,0] = L[1,0]
            Linv[0,1] = L[0,0]
            Linv[1,0] = L[1,0]
            Linv[1,1] = -L[0,0]
            
            if self.AdjointMedium is True:
                if self.verbose is True:
                    print('\nL_eigenvectors_p_w (AdjointMedium is True) and (ReciprocalMedium is False)')
                    print('----------------------------------------------------------------------------')
                    print('For non-reciprocal media, the eigenvector matrix of a medium and its adjoint medium are different.\n')
                
                # L matrix (adjoint medium) = N Transpose( Inverse( L(-p1) )) N
                La = np.zeros((2,2),dtype=complex)
                La[0,0] = (beta/p3n)**0.5
                La[0,1] = La[0,0]
                La[1,0] = (p3n/beta)**0.5
                La[1,1] = -La[1,0]
                La = La/2**0.5
                
                #  Inverse L matrix (adjoint medium) = N Transpose( L(-p1)) N
                Lainv = np.zeros((2,2),dtype=complex)
                Lainv[0,0] = (p3n/beta)**0.5
                Lainv[0,1] = (beta/p3n)**0.5
                Lainv[1,0] = Lainv[0,0]
                Lainv[1,1] = -Lainv[0,1]
                Lainv = Lainv/2**0.5
            
        elif (self.ReciprocalMedium is False) and (normalisation is 'pressure'):
            # L matrix
            L[0,0] = 1
            L[0,1] = 1
            L[1,0] = p3/beta
            L[1,1] = -p3/beta
            
            # Inverse L matrix
            Linv[0,0] = 1
            Linv[0,1] = beta/p3
            Linv[1,0] = 1
            Linv[1,1] = -beta/p3
            Linv = 0.5*Linv
            
            if self.verbose is True and self.AdjointMedium is True:
                print('\nL_eigenvectors_p_w (AdjointMedium is True) and (normalisation=\'pressure\')')
                print('-----------------------------------------------------------------------------')
                print('We have defined the eigenvector matrix of the adjoint medium \'La\' and its inverse \'Lainv\' only for flux-normalisation.\n')
        
        out = {'L':L,'Linv':Linv,'La':La,'Lainv':Lainv}
        return out
          
    def RT_p_w(self,beta_u=None,g3_u=None,p3_u=None,p3n_u=None,beta_l=None,g3_l=None,p3_l=None,p3n_l=None,normalisation='flux'):
        """computes the scattering coefficients at an horizontal interface, either in flux- or in pressure-normalisation. The variables with subscript 'u' refer to the medium parameters in the upper half-space, the variables with subscript 'l' refer to the medium parameters in the lower half-space. We consider a single horizontal ray-parameter :math:`p_1`, which is associated with a vertical ray-parameter 'p3_u' in the upper half-space and 'p3_l' in the lower half-space. If one sets \'AdjointMedium=True\', **RT_p_w** also computes the scattering coefficients in the adjoint medium.
        
        Parameters
        ----------
    
        beta_u : int, float
            Medium parameter :math:`\\beta` (real-valued) (upper half-space).
        
        g3_u : int, float
            Medium parameter :math:`\gamma_3` (upper half-space).
        
        p3_u : int, float
            Vertical ray-parameter :math:`p_3` for a positive horizontal ray-parameter :math:`p_1` (upper half-space).
        
        p3n_u : int, float, optional (required if 'AdjointMedium=True')
            Vertical ray-parameter :math:`p_3` for a negative horizontal ray-parameter :math:`p_1` (upper half-space).
            
        beta_l : int, float
            Medium parameter :math:`\\beta` (real-valued) (lower half-space).
        
        g3_l : int, float
            Medium parameter :math:`\gamma_3` (lower half-space).
        
        p3_l : int, float
            Vertical ray-parameter :math:`p_3` for a positive horizontal ray-parameter :math:`p_1` (lower half-space).
        
        p3n_l : int, float, optional (required if 'AdjointMedium=True')
            Vertical ray-parameter :math:`p_3` for a negative horizontal ray-parameter :math:`p_1` (lower half-space).
            
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
            All scattering coefficients are stored as scalars.
        
        Notes
        -----
    
        - For reciprocal media, the scattering coefficients of the adjoint medium are identical to the scattering coefficients of the true medium.
        - We have defined the scattering coefficients of the adjoint medium only for flux-normalisation.
            
        References
        ----------
        Kees document as soon as it is published.
        
        Examples
        --------

        >>> from Layered_NRM_p_w import Layered_NRM_p_w as LM
        >>> import numpy as np
        
        >>> # Create wavefield F for positive horizontal ray-parameter p1
        >>> F=LM(nt=1024,dt=0.005,x3vec=np.array([1.1,2.2,3.7]),avec=np.array([1,2,3]),bvec=np.array([0.4,3.14,2]),g1vec=np.array([0.9,2.1,0.3]),g3vec=np.array([0.7,1.14,0.2]),p1=2e-4,ReciprocalMedium=False,AdjointMedium=True)
        >>> ScatCoeffs = F.RT_p_w(beta_u=F.bvec[0],g3_u=F.g3vec[0],p3_u=F.p3[0],p3n_u=F.p3n[0],beta_l=F.bvec[1],g3_l=F.g3vec[1],p3_l=F.p3[1],p3n_l=F.p3n[1],normalisation='flux')
        >>> rplus = ScatCoeffs['rP']
        (0.8620013269525346+0.5069060192304582j)
        
        >>> # Create wavefield Fn for negative horizontal ray-parameter p1
        >>> Fn=LM(nt=1024,dt=0.005,x3vec=np.array([1.1,2.2,3.7]),avec=np.array([1,2,3]),bvec=np.array([0.4,3.14,2]),g1vec=np.array([0.9,2.1,0.3]),g3vec=np.array([0.7,1.14,0.2]),p1=-2e-4,ReciprocalMedium=False,AdjointMedium=True)
        >>> ScatCoeffsn = Fn.RT_p_w(beta_u=Fn.bvec[0],g3_u=Fn.g3vec[0],p3_u=Fn.p3[0],p3n_u=Fn.p3n[0],beta_l=Fn.bvec[1],g3_l=Fn.g3vec[1],p3_l=Fn.p3[1],p3n_l=Fn.p3n[1],normalisation='flux')
        
        >>> # In non-reciprocal media, for flux-normalisation, the reflection coefficients
        >>> # in true medium for positive horizontal ray-parameter p1
        >>> # and in adjoint medium for negative horizontal ray-parameter p1
        >>> # are identical:
        >>> np.abs(ScatCoeffs['rP']-ScatCoeffsn['rPa'])
        0.0
        
        """
        
        # Check if required input variables are given
        if (beta_u is None) or (g3_u is None) or (p3_u is None) or (beta_l is None) or (g3_l is None) or (p3_l is None):
            sys.exit('RT_p_w: The input variables \'beta_u\', \'g3_u\',  \'p3_u\', \'beta_l\', \'g3_l\',  \'p3_l\' must be set.')
            
        # Check if normalisation is set correctly
        if (normalisation is not 'flux') and (normalisation is not 'pressure'):
            sys.exit('RT_p_w: The input variable \'normalisation\' must be set, either to \'flux\', or to \'pressure\'.')
            
        # Check if the vertical ray-parameter for a negative horizontal ray-parameter is given
        if  (self.AdjointMedium is True) and ((p3n_u is None) or (p3n_u is None)) and (normalisation is 'flux') and (self.ReciprocalMedium is False):
            sys.exit('RT_p_w: The input variables \'p3n_u\' and \'p3n_l\' (vertical ray-parameter p3 for a negative horizontal ray-parameter p1) must be set to compute the scattering coefficients in the adjoint medium \'rPa\', \'tPa\', \'rMa\' and \'tMa\'.')
            
        # Initialise scattering coefficients in adjoint medium    
        rPa = None
        tPa = None
        rMa = None
        tMa = None
            
        if (self.ReciprocalMedium is True) and (normalisation is 'flux'):
            
            # True medium
            rP =  ( (p3_u+g3_u)*beta_l - (p3_l+g3_l)*beta_u ) / ( (p3_u-g3_u)*beta_l + (p3_l+g3_l)*beta_u )
            rM = -( (p3_u-g3_u)*beta_l - (p3_l-g3_l)*beta_u ) / ( (p3_u-g3_u)*beta_l + (p3_l+g3_l)*beta_u )
            tP = 2*(p3_u*beta_l*p3_l*beta_u)**0.5 / ( (p3_u-g3_u)*beta_l + (p3_l+g3_l)*beta_u )
            tM = tP
            
            if self.AdjointMedium is True:
                # Adjoint medium
                rPa = rP 
                tPa = tP 
                rMa = rM 
                tMa = tM 
    
                if self.verbose is True:
                    print('\nRT_p_w: (AdjointMedium is True) and (ReciprocalMedium is True)')
                    print('----------------------------------------------------------------')
                    print('For reciprocal media, the scattering coefficients in a medium and its adjoint medium are identical.\n')
        
        elif (self.ReciprocalMedium is True) and (normalisation is 'pressure'):
            
            # True medium
            rP =  ( (p3_u+g3_u)*beta_l - (p3_l+g3_l)*beta_u ) / ( (p3_u-g3_u)*beta_l + (p3_l+g3_l)*beta_u )
            tP = 2*p3_u*beta_l / ( (p3_u-g3_u)*beta_l + (p3_l+g3_l)*beta_u )
            rM = -( (p3_u-g3_u)*beta_l - (p3_l-g3_l)*beta_u ) / ( (p3_u-g3_u)*beta_l + (p3_l+g3_l)*beta_u )
            tM = 2*p3_l*beta_u / ( (p3_u-g3_u)*beta_l + (p3_l+g3_l)*beta_u )
            
            if self.verbose is True and self.AdjointMedium is True:
                print('\nRT_p_w: (AdjointMedium is True) and (normalisation=\'pressure\')')
                print('------------------------------------------------------------------')
                print('We have defined the scattering coefficients of the adjoint medium only for flux-normalisation.\n')
            
        elif (self.ReciprocalMedium is False) and (normalisation is 'flux'):
            
            # True medium
            rP = (p3_u*beta_l - p3_l*beta_u) / (p3_u*beta_l + p3_l*beta_u)
            tP = 2*(p3_u*beta_l*p3_l*beta_u)**0.5 / (p3_u*beta_l + p3_l*beta_u)
            rM = -rP
            tM = tP
            
            if self.AdjointMedium is True:
                # Adjoint medium
                rPa = (p3n_u*beta_l - p3n_l*beta_u) / (p3n_u*beta_l + p3n_l*beta_u)
                tPa = 2*(p3n_u*beta_l*p3n_l*beta_u)**0.5 / (p3n_u*beta_l + p3n_l*beta_u)
                rMa = -rPa
                tMa = tPa 
            
        elif (self.ReciprocalMedium is False) and (normalisation is 'pressure'):
            rP = (p3_u*beta_l - p3_l*beta_u) / (p3_u*beta_l + p3_l*beta_u)
            tP = 2*p3_u*beta_l / (p3_u*beta_l + p3_l*beta_u)
            rM = -rP
            tM = 2*p3_l*beta_u / (p3_u*beta_l + p3_l*beta_u)
            
            if self.verbose is True and self.AdjointMedium is True:
                print('\nRT_p_w: (AdjointMedium is True) and (normalisation=\'pressure\')')
                print('------------------------------------------------------------------')
                print('We have defined the scattering coefficients of the adjoint medium only for flux-normalisation.\n')
            
        out = {'rP':rP,'tP':tP,'rM':rM,'tM':tM,'rPa':rPa,'tPa':tPa,'rMa':rMa,'tMa':tMa}
        return out
    
    def W_propagators_p_w(self,p3=None,p3n=None,g3=None,dx3=None,w=None):
        """computes the downgoing propagator 'wP' and the upgoing progagator 'wM' for a single vertical ray-parameter 'p3' and a vertical distance 'dx3' (downward pointing :math:`x_3`-axis).
        
        
        Parameters
        ----------
    
        p3 : int, float
            Vertical ray-parameter :math:`p_3` for a positive horizontal ray-parameter :math:`p_1`.
        
        p3n : int, float, optional (required if 'AdjointMedium=True')
            Vertical ray-parameter :math:`p_3` for a negative horizontal ray-parameter :math:`p_1`.
            
        g3 : int, float
            Medium parameter :math:`\gamma_3`.
            
        dx3 : int, float
            Vertical propagation distance :math:`\Delta x_3` (downward pointing :math:`x_3`-axis).
            
        w : int, float
            Frequency :math:`\omega` in radians. By default the propagators are computed for all sampled (positive) frequencies.
        
            
        Returns
        -------
    
        dict
            Dictionary that contains 
                - **wP**: Downward propagator :math:`\\tilde{w}^+`.
                - **wM**: Upward propagator :math:`\\tilde{w}^-`.
                - **wPa**: Downward propagator :math:`\\tilde{w}^{+(a)}` (adjoint medium). 
                - **wMa**: Upward propagator :math:`\\tilde{w}^{-(a)}` (adjoint medium). 
            All propagators are stored either in an arrays of shape (nf,1), or as a scalar (if the variable'w' is set). The variables 'wPa' and 'wMa' are computed only if one sets 'AdjointMedium=True'.
        
        
        References
        ----------
        Kees document as soon as it is published.


        Examples
        --------

        >>> from Layered_NRM_p_w import Layered_NRM_p_w as LM
        >>> import numpy as np
        
        >>> F=LM(nt=1024,dt=0.005,x3vec=np.array([1.1,2.2,3.7]),avec=np.array([1,2,3]),bvec=np.array([0.4,3.14,2]),g1vec=np.array([0.9,2.1,0.3]),g3vec=np.array([0.7,1.14,0.2]),p1=2e-4,ReciprocalMedium=False)
        >>> W=F.W_propagators_p_w( p3=F.p3[0] , p3n=F.p3n[0] , g3=F.g3vec[0] , dx3=F.x3vec[1]-F.x3vec[0])
        
        >>> W['wP']
        [[ 1.00000000e+00+0.        j],[0.24690276+0.34159244j],...,[1.07001473e-192-1.48037608e-192j]]
        
        >>> # The propagator has nf samples because it is computed only for positive frequencies
        >>> W['wP'].shape
        (513, 1)
        
        >>> # AdjointMedium=False, hence, we expect output 'None'
        >>> W['wPa']
       
       
        """
        
        # Check if required input variables are given
        if (not np.isscalar(p3)) or (not np.isscalar(g3)) or (not np.isscalar(dx3)):
            sys.exit('W_propagators_p_w: The input variables \'p3\', \'g3\' and  \'dx3\' must be set as scalars.')
            
        # If AdjointMedium=True it is required to set p3n=p3(-p1)
        if (self.AdjointMedium is True) and (not np.isscalar(p3n)):
            sys.exit('W_propagators_p_w: If AdjointMedium=True the input variable \'p3n\' must be set as a scalar.')
        
        # Make frequency vector
        if w is None:
            w = self.Wvec()
        # If frequency is given check if it is a scalar
        elif not np.isscalar(w):
            sys.exit('W_propagators_p_w: If \'w\' is set it must be a scalar.')   
        elif w.real < 0:
            sys.exit('W_propagators_p_w: The real-part of \'w\' must not be smaller than 0.') 
        else:
            # We define w as an array to be able to use .copy()
            w = np.array([[w]])
        
        
        if self.ReciprocalMedium is True:
            
            wP = np.exp(1j*w*p3*dx3)
            wM = wP.copy()
            
            if self.AdjointMedium is True:
                wPa = np.exp(1j*w*p3n*dx3)
                wMa = wPa.copy()
            else: 
                wPa = None
                wMa = None
        
        elif self.ReciprocalMedium is False:
            
            wP = np.exp(1j*w*(p3+g3)*dx3)
            wM = np.exp(1j*w*(p3-g3)*dx3)
        
            if self.AdjointMedium is True:
                wPa = np.exp(1j*w*(p3n-g3)*dx3)
                wMa = np.exp(1j*w*(p3n+g3)*dx3)
            else: 
                wPa = None
                wMa = None
                
        out = {'wP':wP,'wM':wM,'wPa':wPa,'wMa':wMa}
        return out
    
    def RT_response_p_w(self,x3vec=None,avec=None,bvec=None,g1vec=None,g3vec=None,normalisation='flux',InternalMultiples=True):
        """computes the reflection and transmission responses from above and from below. If medium parameters are given the computed responses are associated with the given parameters medium. Otherwise, the medium parameters defined in **Layered_NRM_p_w** are used.
        
        The medium responses are associated to measurements at :math:`x_3=0` and at :math:`x_3=` 'x3vec[-2]' :math:`+\epsilon`, where :math:`\epsilon` is an infinitesimally small positive constant. Hence, the propagation from :math:`x_3=0` to the shallowest interface is included. However, the propagation through the deepest layer is excluded.
        
        Parameters
        ----------
    
        x3vec : numpy.ndarray, optional
            Vertical spatial vector :math:`x_3`, for n layers 'x3vec' must have the shape (n,). We define the :math:`x_3`-axis as downward-pointing. Implicitly, the first value on the :math:`x_3`-axis is zero (not stored in 'x3vec').
    
        avec : numpy.ndarray, optional
            Medium parameter :math:`\\alpha` (real-valued), for n layers 'avec' must have the shape (n,).
            
        bvec : numpy.ndarray, optional
            Medium parameter :math:`\\beta` (real-valued), for n layers 'bvec' must have the shape (n,).
        
        g1vec : numpy.ndarray, optional
            Medium parameter :math:`\gamma_1` (real-valued for non-reciprocal media or imaginary-valued for reciprocal media), for n layers 'g1vec' must have the shape (n,).
            
        g3vec : numpy.ndarray, optional
            Medium parameter :math:`\gamma_3` (real-valued for non-reciprocal media or imaginary-valued for reciprocal media), for n layers 'g3vec' must have the shape (n,).
            
        normalisation : str, optional
            For pressure-normalisation set normalisation='pressure', for flux-normalisation set normalisation='flux'.
            
        InternalMultiples : bool, optional
            To model internal multiples set 'InternalMultiples=True'. To ignore internal multiples set 'InternalMultiples=False'.
            
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
            All medium responses are stored in arrays of shape (nf,1). The variables 'RPa', 'TPa', 'RMa' and 'TMa' are computed only if one sets 'AdjointMedium=True'.
        
        
        References
        ----------
        Kees document as soon as it is published.
        
        Examples
        --------

        >>> from Layered_NRM_p_w import Layered_NRM_p_w as LM
        >>> import numpy as np
        
        >>> # Initialise a wavefield in a 1D reciprocal medium
        >>> # Here, the parameters are chosen such that the wavefield is purely propagating (not evanescent)
        >>> F=LM(nt=1024,dt=0.005,x3vec=np.array([100,500,1000,1010]),avec=np.array([5,2,3,4]),bvec=np.array([0.4,3.14,2,1.5]),g1vec=np.array([0.9,2.1,0.3,0.25]),g3vec=np.array([0.7,1.14,0.2,0.3]),p1=2e-4,ReciprocalMedium=False)
        
        >>> # Model the medium responses
        >>> Responses=F.RT_response_p_w(normalisation='flux',InternalMultiples=True)
        >>> # Here are your first medium responses:
        >>> Rplus = Responses['RP']
        >>> Tplus = Responses['TP']
        
        >>> # Verify if conservation of energy is satisfied
        >>> Rplus.conj()*Rplus+Tplus.conj()*Tplus
        ([[1.+0.j], [1.+0.j], ..., [1.+0.j]])
        
        
        """
        
        # Check if normalisation is set correctly
        if (normalisation is not 'flux') and (normalisation is not 'pressure'):
            sys.exit('RT_response_p_w: The input variable \'normalisation\' must be set, either to \'flux\', or to \'pressure\'.')
        # Medium responses of the adjoint medium can only be computed in flux-normalisation
        if (self.AdjointMedium is True) and (normalisation is 'pressure'):
            sys.exit('RT_response_p_w: We have defined the scattering coefficients of the adjoint medium only for flux-normalisation.')
        
        # Check if a layer stack is given
        if isinstance(x3vec,np.ndarray) and isinstance(avec,np.ndarray) \
        and isinstance(bvec,np.ndarray) and isinstance(g1vec,np.ndarray) \
        and isinstance(g3vec,np.ndarray):
            
            # Create a wavefield in a sub-medium
            # I do this because when the sub-wavefield is initialised all parameters 
            # are automatically tested for correctness
            self.SubSelf = Layered_NRM_p_w(self.nt,self.dt,self.nr,self.dx1,self.verbose,x3vec=x3vec,avec=avec,bvec=bvec,\
                                           g1vec=g1vec,g3vec=g3vec,p1=self.p1,ReciprocalMedium=self.ReciprocalMedium,\
                                           AdjointMedium=self.AdjointMedium)
            
            x3vec = self.SubSelf.x3vec
            bvec = self.SubSelf.bvec
            g3vec = self.SubSelf.g3vec
            p3 = self.SubSelf.p3
            p3n = self.SubSelf.p3n
        
            del self.Subself
                
        # Else compute response of entire medium
        else:
            x3vec = self.x3vec
            bvec = self.bvec
            g3vec = self.g3vec
            p3 = self.p3
            p3n = self.p3n
            
        
        # Number of layers
        N = np.size(x3vec)
        
        # Vector with layer thicknesses
        dx3vec = x3vec.copy()
        dx3vec[1:] = x3vec[1:]-x3vec[:-1]
        
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
            RPa = RP.copy()
            RMa = RP.copy()
            TPa = TP.copy()
            TMa = TP.copy()
        else:
            RPa = None
            TPa = None
            RMa = None
            TMa = None
        
        # Loop over N-1 interfaces
        for n in range(0,N-1):
            
            # Scattering coefficients
            ScatCoeffs = self.RT_p_w(beta_u=bvec[n]  ,g3_u=g3vec[n]  ,p3_u=p3[n]  ,p3n_u=p3n[n],\
                                     beta_l=bvec[n+1],g3_l=g3vec[n+1],p3_l=p3[n+1],p3n_l=p3n[n+1],\
                                     normalisation=normalisation)
            
            rP = ScatCoeffs['rP']
            tP = ScatCoeffs['tP']
            rM = ScatCoeffs['rM']
            tM = ScatCoeffs['tM']
            
            # Propagators
            W = self.W_propagators_p_w(p3=p3[n],p3n=p3n[n],g3=g3vec[n],dx3=dx3vec[n])
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
            
            if (np.isnan(RP).any() or np.isnan(TP).any() or np.isnan(RM).any() or np.isnan(TM).any()\
                or np.isinf(RP).any() or np.isinf(TP).any() or np.isinf(RM).any() or np.isinf(TM).any()):
                print('\n')
                print('Layered_NRM_p_w:')
                print('\n'+100*'-'+'\n')
                print('One of the modelled wavefields in the true medium contains a NaN (Not a Number) or an Inf (infinite) element.')
                print('\n')
                
                if np.isnan(RP).any():
                    print('\t - RP contains '+np.count_nonzero(np.isnan(RP))+' NaN.')
                if np.isinf(RP).any():
                    print('\t - RP contains '+np.count_nonzero(np.isinf(RP))+' Inf.')
                if np.isnan(TP).any():
                    print('\t - TP contains '+np.count_nonzero(np.isnan(TP))+' NaN.')
                if np.isinf(TP).any():
                    print('\t - TP contains '+np.count_nonzero(np.isinf(TP))+' Inf.')
                if np.isnan(RM).any():
                    print('\t - RM contains '+np.count_nonzero(np.isnan(RM))+' NaN.')
                if np.isinf(RM).any():
                    print('\t - RM contains '+np.count_nonzero(np.isinf(RM))+' Inf.')
                if np.isnan(TM).any():
                    print('\t - TM contains '+np.count_nonzero(np.isnan(TM))+' NaN.')
                if np.isinf(TM).any():
                    print('\t - TM contains '+np.count_nonzero(np.isinf(TM))+' Inf.')
            
            if self.AdjointMedium is True:
                
                if (np.isnan(RPa).any() or np.isnan(TPa).any() or np.isnan(RMa).any() or np.isnan(TMa).any()\
                or np.isinf(RPa).any() or np.isinf(TPa).any() or np.isinf(RMa).any() or np.isinf(TMa).any()):
                    print('\n')
                    print('Layered_NRM_p_w:')
                    print('\n'+100*'-'+'\n')
                    print('One of the modelled wavefields in the adoint medium contains a NaN (Not a Number) or an Inf (infinite) element.')
                    print('\n')
                    
                    if np.isnan(RPa).any():
                        print('\t - RPa contains '+np.count_nonzero(np.isnan(RPa))+' NaN.')
                    if np.isinf(RPa).any():
                        print('\t - RPa contains '+np.count_nonzero(np.isinf(RPa))+' Inf.')
                    if np.isnan(TPa).any():
                        print('\t - TPa contains '+np.count_nonzero(np.isnan(TPa))+' NaN.')
                    if np.isinf(TPa).any():
                        print('\t - TPa contains '+np.count_nonzero(np.isinf(TPa))+' Inf.')
                    if np.isnan(RMa).any():
                        print('\t - RMa contains '+np.count_nonzero(np.isnan(RMa))+' NaN.')
                    if np.isinf(RMa).any():
                        print('\t - RMa contains '+np.count_nonzero(np.isinf(RMa))+' Inf.')
                    if np.isnan(TMa).any():
                        print('\t - TMa contains '+np.count_nonzero(np.isnan(TMa))+' NaN.')
                    if np.isinf(TMa).any():
                        print('\t - TMa contains '+np.count_nonzero(np.isinf(TMa))+' Inf.')
                
                print('\n')

                
        out={'RP':RP,'TP':TP,'RM':RM,'TM':TM,'RPa':RPa,'TPa':TPa,'RMa':RMa,'TMa':TMa}
        return out
            